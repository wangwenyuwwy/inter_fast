import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class OptimizedHeterogeneousBranch(nn.Module):
  def __init__(self, in_dim, out_dim, qp_types):
    super().__init__()

    # 第一阶段：使用AGFB替代原有双分支
    self.stage1 = nn.Sequential(
      AGFB(in_dim, heads=2),
      nn.GELU(),
      AGFB(in_dim, heads=4)
    )

    # 第二阶段：轻量级时空注意力
    self.stage2 = nn.Sequential(
      nn.Conv2d(in_dim, in_dim * 2, 3, stride=2, padding=1),  # 下采样
      nn.GELU(),
      nn.Conv2d(in_dim * 2, in_dim, 1),
      SimplifiedAttention(in_dim, qp_types)  # 简化注意力模块
    )

    # 动态预测头
    self.head = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      nn.Linear(in_dim, 256),
      nn.LayerNorm(256),
      nn.GELU(),
      nn.Linear(256, out_dim)
    )

  def forward(self, x, qp):
    x = self.stage1(x)
    x = self.stage2(x)
    return self.head(x)


class SimplifiedAttention(nn.Module):
  """整合时空注意力的轻量版本"""

  def __init__(self, dim, qp_types):
    super().__init__()
    self.qp_embed = nn.Embedding(qp_types, dim)

    # 联合空间通道注意力
    self.joint_att = nn.Sequential(
      nn.Conv2d(dim, dim // 4, 1),
      nn.GELU(),
      nn.Conv2d(dim // 4, dim, 1),
      nn.Sigmoid()
    )

    # 轻量时序建模
    self.temp_att = nn.Sequential(
      nn.Conv1d(dim, dim // 2, 3, padding=1),
      nn.GELU(),
      nn.Conv1d(dim // 2, dim, 1),
      nn.Sigmoid()
    )

  def forward(self, x, qp):
    B, C, H, W = x.shape

    # 空间通道联合注意力
    spatial_att = self.joint_att(x)

    # 伪时序注意力
    temporal = x.mean([2, 3])  # [B,C]
    temporal_att = self.temp_att(temporal.unsqueeze(-1))  # [B,C,1]

    # QP条件调制
    qp_emb = self.qp_embed(qp).view(B, C, 1, 1)

    return x * spatial_att * temporal_att.view(B, C, 1, 1) * qp_emb


class AGFB(nn.Module):
  """优化后的轴向门控融合块"""

  def __init__(self, dim, heads=4):
    super().__init__()

    # 轴向注意力分支
    self.axial_path = nn.Sequential(
      AxialAttention(dim, heads),
      nn.Conv2d(dim, dim // 2, 1),
      nn.GELU()
    )

    # 卷积分支
    self.conv_path = nn.Sequential(
      nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
      nn.Conv2d(dim, dim // 2, 1),
      nn.GELU()
    )

    # 动态融合门控
    self.gate = nn.Sequential(
      nn.Conv2d(dim, 2, 1),
      nn.Softmax(dim=1)
    )

    # 通道重校准
    self.se = SqueezeExcitation(dim)

  def forward(self, x):
    axial = self.axial_path(x)
    conv = self.conv_path(x)

    fused = torch.cat([axial, conv], dim=1)
    gates = self.gate(x)  # [B,2,H,W]

    out = gates[:, 0:1] * axial + gates[:, 1:2] * conv
    return self.se(out) + x  # 残差连接

class AxialAttention(nn.Module):
    """轴向注意力（水平与垂直方向分离计算）"""

    def __init__(self, dim, heads=4):
      super().__init__()
      self.heads = heads
      self.dim_head = dim // heads

      # 水平方向注意力
      self.to_qkv_h = nn.Conv2d(dim, dim * 3, 1, bias=False)
      # 垂直方向注意力
      self.to_qkv_v = nn.Conv2d(dim, dim * 3, 1, bias=False)

      self.scale = self.dim_head ** -0.5

    def forward(self, x):
      b, c, h, w = x.shape

      # 水平方向处理
      qkv_h = self.to_qkv_h(x).chunk(3, dim=1)
      q_h, k_h, v_h = map(
        lambda t: t.permute(0, 2, 3, 1).reshape(b * h, w, self.heads, self.dim_head).permute(0, 2, 1, 3), qkv_h)
      attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
      attn_h = attn_h.softmax(dim=-1)
      out_h = (attn_h @ v_h).permute(0, 2, 1, 3).reshape(b, h, w, c).permute(0, 3, 1, 2)

      # 垂直方向处理
      qkv_v = self.to_qkv_v(x).chunk(3, dim=1)
      q_v, k_v, v_v = map(
        lambda t: t.permute(0, 3, 2, 1).reshape(b * w, h, self.heads, self.dim_head).permute(0, 2, 1, 3), qkv_v)
      attn_v = (q_v @ k_v.transpose(-2, -1)) * self.scale
      attn_v = attn_v.softmax(dim=-1)
      out_v = (attn_v @ v_v).permute(0, 2, 1, 3).reshape(b, w, h, c).permute(0, 3, 2, 1)

      return (out_h + out_v) * 0.5


class SqueezeExcitation(nn.Module):
  """通道注意力模块"""

  def __init__(self, channel, reduction=4):
    super().__init__()
    self.avgpool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(channel, channel // reduction),
      nn.ReLU(inplace=True),
      nn.Linear(channel // reduction, channel),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avgpool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y.expand_as(x)