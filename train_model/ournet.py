import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# 核心模块定义
# --------------------------

class GhostModule(nn.Module):
  """Ghost卷积模块 (参数减少版)"""

  def __init__(self, in_ch, out_ch, kernel_size=1, ratio=2):
    super().__init__()
    self.primary = nn.Sequential(
      nn.Conv2d(in_ch, out_ch // ratio, kernel_size, padding=kernel_size // 2, bias=False),
      nn.BatchNorm2d(out_ch // ratio),
      nn.ReLU6(inplace=True)
    )
    self.cheap = nn.Sequential(
      nn.Conv2d(out_ch // ratio, out_ch // ratio, 3, padding=1, groups=out_ch // ratio, bias=False),
      nn.BatchNorm2d(out_ch // ratio),
      nn.ReLU6(inplace=True)
    )

  def forward(self, x):
    x1 = self.primary(x)
    x2 = self.cheap(x1)
    return torch.cat([x1, x2], dim=1)


class DynamicSparseRouter(nn.Module):
  """动态稀疏路由层"""

  def __init__(self, in_ch, out_ch, threshold=0.5):
    super().__init__()
    self.threshold = threshold
    # 复杂路径
    self.complex_path = nn.Sequential(
      GhostModule(in_ch, out_ch),
      nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
      nn.BatchNorm2d(out_ch),
      nn.ReLU6()
    )
    # 简单路径
    self.simple_path = nn.Conv2d(in_ch, out_ch, 1)

  def forward(self, x):
    # 计算梯度能量 (近似实现)
    grad_x = F.conv2d(x, torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32, device=x.device))
    grad_y = F.conv2d(x, torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32, device=x.device))
    energy = (grad_x.abs() + grad_y.abs()).mean(dim=[1, 2, 3], keepdim=True)

    # 路由选择 (Straight-Through Estimator)
    mask = (energy > self.threshold).float()
    x_complex = self.complex_path(x)
    x_simple = self.simple_path(x)

    # 合并路径
    return mask * x_complex + (1 - mask) * x_simple + \
           (x_complex - x_complex.detach()) * (1 - mask) + \
           (x_simple - x_simple.detach()) * mask  # STE梯度保持


class RecursivePartitionHead(nn.Module):
  """递归划分预测头"""

  def __init__(self, feat_ch, num_classes, min_cu_size=8):
    super().__init__()
    self.min_cu_size = min_cu_size
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Linear(feat_ch, num_classes)  # 权重共享

  def forward(self, feats, current_size):
    """
    feats: 当前CU的特征图 [B, C, H, W]
    current_size: 当前CU的尺寸 (如64)
    """
    if current_size <= self.min_cu_size:
      return None  # 递归终止

    # 全局特征提取
    x = self.pool(feats).squeeze(-1).squeeze(-1)
    logits = self.fc(x)  # [B, num_classes]

    # 递归子CU预测
    child_probs = []
    if current_size > self.min_cu_size:
      # 划分四子CU (示例实现)
      h, w = feats.size()[2:]
      sub_feats = [feats[:, :, :h // 2, :w // 2],
                   feats[:, :, :h // 2, w // 2:],
                   feats[:, :, h // 2:, :w // 2],
                   feats[:, :, h // 2:, w // 2:]]

      for sf in sub_feats:
        child_logits = self.forward(sf, current_size // 2)
        if child_logits is not None:
          child_probs.append(F.softmax(child_logits, dim=1))

    return {'logits': logits, 'children': child_probs}


# --------------------------
# 完整网络架构
# --------------------------

class DSCPNet(nn.Module):
  def __init__(self, in_ch=3, base_ch=16, num_classes=5):
    super().__init__()
    # 主干网络 (简化的MobileNetV3-Ghost)
    self.backbone = nn.Sequential(
      nn.Conv2d(in_ch, base_ch, 3, stride=2, padding=1),
      nn.BatchNorm2d(base_ch),
      nn.ReLU6(),

      DynamicSparseRouter(base_ch, base_ch * 2),
      GhostModule(base_ch * 2, base_ch * 4),

      nn.MaxPool2d(2),
      DynamicSparseRouter(base_ch * 4, base_ch * 8),
    )

    # 多尺度特征提取
    self.pyramid = nn.ModuleDict({
      '64': nn.Sequential(
        DynamicSparseRouter(base_ch * 8, base_ch * 16),
        nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1)
      ),
      '32': nn.Sequential(
        nn.MaxPool2d(2),
        DynamicSparseRouter(base_ch * 16, base_ch * 32)
      )
    })

    # 递归预测头
    self.head = RecursivePartitionHead(base_ch * 32, num_classes)

  def forward(self, x):
    # 特征提取
    x = self.backbone(x)  # [B, C, 32, 32]
    p64 = self.pyramid['64'](x)
    p32 = self.pyramid['32'](p64)

    # 递归预测
    return self.head(p32, current_size=32)  # 假设输入为128x128，此时p32对应32x32 CU


# --------------------------
# 训练示例
# --------------------------

if __name__ == '__main__':
  # 模型初始化
  model = DSCPNet()
  input_tensor = torch.randn(2, 3, 128, 128)

  # 前向传播
  output = model(input_tensor)


  # 损失计算示例 (需根据实际标签结构调整)
  def recursive_loss(output, target):
    loss = F.cross_entropy(output['logits'], target['label'])
    for i, child in enumerate(output['children']):
      loss += recursive_loss(child, target['children'][i])
    return loss


  # 训练循环框架
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
  for epoch in range(100):
    # 假设data_loader为自定义数据加载器
    for data, target in data_loader:
      pred = model(data)
      loss = recursive_loss(pred, target)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()