import numpy as np
import torch
import torch.nn as nn
import math
import pywt
import torch.nn.functional as F
import time
import cv2



class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, padding, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out




def PywtTransfrom(x):
    ca, (ch, cv, cd) = pywt.dwt2(x, 'haar')
    return ca, ch, cv, cd


class MT_Net(nn.Module):  # luma bt depth and direction prediction
    def __init__(self):
        super(MT_Net, self).__init__()
        self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
        self.padding_rb = nn.ZeroPad2d((0, 4, 0, 4))
        self.padding_r = nn.ZeroPad2d((0, 4, 0, 0))
        self.padding_b = nn.ZeroPad2d((0, 0, 0, 4))

        self.conv_b1_1 = nn.Conv2d(1, 16, kernel_size=(9, 9), padding=0, stride=1)
        self.conv_b1_2 = nn.Conv2d(1, 8, kernel_size=(5, 9), padding=0, stride=1)
        self.conv_b1_3 = nn.Conv2d(1, 8, kernel_size=(9, 5), padding=0, stride=1)
        # M-Main, B-Branch, A-Attention
        self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1))
        self.trunk_M2 = nn.Sequential(ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 64, 3, 1))
        self.trunk_B1 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.trunk_B2 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.trunk_B3 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.conv_B1 = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)
        self.conv_B2 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.conv_B3 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        # self.resblock_A1 = ResidualBlock(2, 64, kernel_size=3, padding=1)
        # self.resblock_A2 = ResidualBlock(2, 64, kernel_size=3, padding=1)
        self.trunk_Att1 = nn.Sequential(ResidualBlock(3, 32, 3, 1), ResidualBlock(32, 64, 3, 1))
        self.trunk_Att2 = nn.Sequential(ResidualBlock(3, 32, 3, 1), ResidualBlock(32, 64, 3, 1))


    def forward(self, x1):
        x2 = self.padding_lu(x1)  # 1*68*68
        x3_1 = F.relu(self.conv_b1_1(self.padding_rb(x2)))  # 16*64*64
        x3_2 = F.relu(self.conv_b1_2(self.padding_r(x2)))  # 8*64*64
        x3_3 = F.relu(self.conv_b1_3(self.padding_b(x2)))  # 8*64*64
        x3 = torch.cat([x3_1, x3_2, x3_3], 1)  # 32*64*64
        x4 = F.max_pool2d(self.trunk_M1(x3), 2)  # 64*32*32 M1 out
        x5 = F.max_pool2d(self.trunk_M2(x4), 2)  # 64*16*16 M2 out
        x6 = self.trunk_B1(x5)  # 8*16*16
        out0 = self.conv_B1(x6)  # 1*16*16
        return out0


class Luma_Q_Net(nn.Module):  # luma QT depth prediction
    def __init__(self):
        super(Luma_Q_Net, self).__init__()

        self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
        self.padding_rb = nn.ZeroPad2d((0, 4, 0, 4))
        self.padding_r = nn.ZeroPad2d((0, 4, 0, 0))
        self.padding_b = nn.ZeroPad2d((0, 0, 0, 4))
        self.padding_luma = nn.ZeroPad2d((4, 4, 4, 4))

        self.conv_q1 = nn.Conv2d(1, 32, kernel_size=9, padding=0, stride=1)
        self.resblock_q1 = ResidualBlock(32, 64, kernel_size=5, padding=2)
        self.resblock_q2 = ResidualBlock(64, 64, kernel_size=5, padding=2)
        self.resblock_q3 = ResidualBlock(64, 32, kernel_size=3, padding=1)
        # multi pooling
        self.resblock_q4 = ResidualBlock(128, 32, kernel_size=3, padding=1)
        self.resblock_q5 = ResidualBlock(32, 32, kernel_size=3, padding=1)
        self.resblock_q6 = ResidualBlock(32, 8, kernel_size=3, padding=1)
        self.conv_q2 = nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, x):  # input 1*64*64
        x1 = self.padding_luma(x)  # 1*72*72
        x2 = F.relu(self.conv_q1(x1))  # 32*64*64
        x3 = F.max_pool2d(self.resblock_q1(x2), 2)  # 64*32*32
        x4 = F.max_pool2d(self.resblock_q2(x3), 2)  # 64*16*16

        x5 = self.resblock_q3(x4)  # 32*16*16
        x5_1 = F.interpolate(F.max_pool2d(x5, 2), scale_factor=2)
        x5_2 = F.interpolate(F.max_pool2d(x5, 4), scale_factor=4)
        x5_3 = F.interpolate(F.max_pool2d(x5, 8), scale_factor=8)
        x6 = torch.cat([x5, x5_1, x5_2, x5_3], 1)  # 128*16*16
        x7 = self.resblock_q4(x6)  # 32*16*16
        x8 = F.max_pool2d(self.resblock_q5(x7), 2)  # 32*8*8
        x9 = self.resblock_q6(x8)  # 8*8*8
        x10 = self.conv_q2(x9)  # 1*8*8 qt depth map
        # if add_noise == 1:
        #     xq10 = UniverseQuant.apply(x10)
        # else:
        #     xq10 = x10
        # start bt depth
        return x10  # qt depth map


class Edge_Net(nn.Module):  # luma bt depth and direction prediction
    def __init__(self):
        super(Edge_Net, self).__init__()
        self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
        self.padding_rb = nn.ZeroPad2d((0, 8, 0, 8))
        self.padding_r = nn.ZeroPad2d((4, 4, 2, 2))
        self.padding_b = nn.ZeroPad2d((2, 2, 4, 4))
        self.padding_luma = nn.ZeroPad2d((1, 1, 1, 1))
        self.padding_luma1 = nn.ZeroPad2d((4, 4, 4, 4))
        self.conv_b1_1 = nn.Conv2d(1, 16, kernel_size=(9, 9), padding=0, stride=1)
        self.conv_b1_2 = nn.Conv2d(1, 8, kernel_size=(5, 9), padding=0, stride=1)
        self.conv_b1_3 = nn.Conv2d(1, 8, kernel_size=(9, 5), padding=0, stride=1)
        self.conv_q1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
        self.resblock_q1 = ResidualBlock(16, 16, kernel_size=5, padding=2)
        self.resblock_q2 = ResidualBlock(32, 16, kernel_size=5, padding=2)

        # M-Main, B-Branch, A-Attention
        self.trunk_M0 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1))
        self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
                                      ResidualBlock(32, 32, 3, 1),ResidualBlock(32, 32, 3, 1))
        self.trunk_M2 = nn.Sequential(ResidualBlock(32, 32, 5, 2), ResidualBlock(32, 32, 5, 2),
                                      ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
                                      ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 16, 3, 1))
        self.trunk_B1 = nn.Sequential(ResidualBlock(16, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.trunk_Att1 = nn.Sequential(ResidualBlock(1, 16, 3, 1), ResidualBlock(16, 32, 3, 1))

    # x0_0 原始像素
    # x0_1 canny点图 1 2
    # def forward(self, x0_0, x0_1):  # 1x64x64, 1x64x64
    #     # x0_2 = F.relu(self.conv_q1(x0_0))  # 8*64*64
    #     # x1_1 = F.interpolate(F.max_pool2d(x0_2, 4), scale_factor=4)
    #     # x1_2 = F.interpolate(F.max_pool2d(x0_2, 8), scale_factor=8)
    #     # x1_3 = F.interpolate(F.max_pool2d(x0_2, 16), scale_factor=16)
    #     x1_1 = F.relu(self.conv_b1_1(self.padding_luma1(x0_0)))  # 16*64*64
    #     x1_2 = F.relu(self.conv_b1_2(self.padding_r(x0_0)))  # 8*64*64
    #     x1_3 = F.relu(self.conv_b1_3(self.padding_b(x0_0)))  # 8*64*64
    #     x2 = torch.cat([x1_1, x1_2, x1_3], 1)  # 32*64*64
    #     x2_1 = F.max_pool2d(self.trunk_M0(x2),2) # 32*32*32
    #
    #     x2_att = F.max_pool2d(self.trunk_Att1(x0_1),2)  # 32x32x32
    #     x3 = x2_1 * x2_att  # 32x32x32
    #     x4 = self.padding_luma(x3) # 32x34x34
    #
    #     # x4 = F.max_pool2d(self.trunk_M1(x3), 2)  # 64*34*34 M1 out
    #     x5 = F.max_pool2d((self.trunk_M1(x4)), 2)  # 32*17*17 M2 out
    #     x6 = self.trunk_M2(x5)  # 16*17*17
    #     x7 = self.trunk_B1(x6)  # 8*17*17
    #     out = self.conv_B1(x7)  # 2*17*17
    #     return out

    def forward(self, x0_0, x0_1):  # 1x64x64, 1x64x64
        # x0_2 = F.relu(self.conv_q1(x0_0))  # 8*64*64
        # x1_1 = F.interpolate(F.max_pool2d(x0_2, 4), scale_factor=4)
        # x1_2 = F.interpolate(F.max_pool2d(x0_2, 8), scale_factor=8)
        # x1_3 = F.interpolate(F.max_pool2d(x0_2, 16), scale_factor=16)
        x1_1 = F.relu(self.conv_b1_1(self.padding_luma1(x0_0)))  # 16*64*64
        x1_2 = F.relu(self.conv_b1_2(self.padding_r(x0_0)))  # 8*64*64
        x1_3 = F.relu(self.conv_b1_3(self.padding_b(x0_0)))  # 8*64*64
        x2 = torch.cat([x1_1, x1_2, x1_3], 1)  # 32*64*64
        x2_1 = F.max_pool2d(self.trunk_M0(x2),2) # 32*32*32

        # x2_att = F.max_pool2d(self.trunk_Att1(x0_1),2)  # 32x32x32
        # x3 = x2_1 * x2_att  # 32x32x32
        x4 = self.padding_luma(x2_1) # 32x34x34

        # x4 = F.max_pool2d(self.trunk_M1(x3), 2)  # 64*34*34 M1 out
        x5 = F.max_pool2d((self.trunk_M1(x4)), 2)  # 32*17*17 M2 out
        x6 = self.trunk_M2(x5)  # 16*17*17
        x7 = self.trunk_B1(x6)  # 8*17*17
        out = self.conv_B1(x7)  # 2*17*17
        return out


class Edge_Net_bxl(nn.Module):  # luma bt depth and direction prediction
  def __init__(self):
    super(Edge_Net_bxl, self).__init__()
    self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
    self.padding_rb = nn.ZeroPad2d((0, 8, 0, 8))
    self.padding_r = nn.ZeroPad2d((4, 4, 2, 2))
    self.padding_b = nn.ZeroPad2d((2, 2, 4, 4))
    self.padding_luma = nn.ZeroPad2d((1, 1, 1, 1))
    self.padding_luma1 = nn.ZeroPad2d((4, 4, 4, 4))
    self.conv_b1_1 = nn.Conv2d(8, 16, kernel_size=(9, 9), padding=0, stride=1)
    self.conv_b1_2 = nn.Conv2d(8, 8, kernel_size=(5, 9), padding=0, stride=1)
    self.conv_b1_3 = nn.Conv2d(8, 8, kernel_size=(9, 5), padding=0, stride=1)
    self.conv_q1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
    self.resblock_q1 = ResidualBlock(16, 16, kernel_size=5, padding=2)
    self.resblock_q2 = ResidualBlock(32, 16, kernel_size=5, padding=2)

    # M-Main, B-Branch, A-Attention
    self.trunk_M0 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1),
                                  ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1))
    self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1),
                                  ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
                                  ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 32, 3, 1))
    self.trunk_M2 = nn.Sequential(ResidualBlock(32, 32, 5, 2), ResidualBlock(32, 32, 5, 2),
                                  ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
                                  ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 16, 3, 1))
    self.trunk_B1 = nn.Sequential(ResidualBlock(16, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
    self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
    self.trunk_Att1 = nn.Sequential(ResidualBlock(1, 16, 3, 1), ResidualBlock(16, 32, 3, 1))

    # Attention modules
    # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
    # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))

    # Final fully connected layers for output matching subnet3
    self.fc1 = nn.Linear(64, 64)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(64, 6)

  def forward(self, x0_0):
    # Apply attention at the input level
    res = x0_0.clone()
    # atten_value_1 = self.atten_module_1[qp_list]
    # res = res * (atten_value_1.view(atten_value_1.shape[0], atten_value_1.shape[1], 1, 1))

    # Perform convolutions and concatenations
    x1_1 = F.relu(self.conv_b1_1(self.padding_luma1(res)))  # 16*64*64
    x1_2 = F.relu(self.conv_b1_2(self.padding_r(res)))  # 8*64*64
    x1_3 = F.relu(self.conv_b1_3(self.padding_b(res)))  # 8*64*64
    x2 = torch.cat([x1_1, x1_2, x1_3], 1)  # 32*64*64

    # Process through trunk modules
    x2_1 = F.max_pool2d(self.trunk_M0(x2), 2)  # 32*32*32
    # x4 = self.padding_luma(x2_1)  # 32x34x34
    x5 = F.max_pool2d(self.trunk_M1(x2_1), 2)  # 32*16*16 M2 out
    x6 = self.trunk_M2(x5)  # 16*16*16

    # Apply attention after processing through trunk modules
    res2 = x6.clone()
    # atten_value_2 = self.atten_module_2[qp_list]
    # res2 = res2 * atten_value_2

    # Flatten the tensor to match the input shape of the fully connected layer
    # res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]

    # Pass through fully connected layers
    # res2 = self.fc1(res2)
    # res2 = self.relu(res2)
    # out = self.fc2(res2)

    return res2

class Sub_Net_bxl_1616(nn.Module):  # luma bt depth and direction prediction
    def __init__(self):
      super(Sub_Net_bxl_1616, self).__init__()

      # Attention modules
      # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
      # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))

      # Final fully connected layers for output matching subnet3
      self.fc1 = nn.Linear(512, 256)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(256, 64)
      self.relu = nn.ReLU()
      self.fc3 = nn.Linear(64, 6)

      self.qp_fc = nn.Linear(1, 64)

    def forward(self, res2, qp):


      # Flatten the tensor to match the input shape of the fully connected layer
      res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]

      # Pass through fully connected layers
      # res2 = self.fc1(res2)
      # res2 = self.relu(res2)
      res2 = self.fc2(res2)

      qp = qp.view(-1, 1).float()
      qp_feat = self.qp_fc(qp)

      res2 = res2 + qp_feat

      res2 = self.relu(res2)
      out = self.fc3(res2)

      return out


# class Sub_Net_bxl_816(nn.Module):  # luma bt depth and direction prediction
#   def __init__(self):
#     super(Sub_Net_bxl_816, self).__init__()
#
#     # Attention modules
#     # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
#     # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))
#
#     # Final fully connected layers for output matching subnet3
#     self.fc1 = nn.Linear(512, 256)
#     self.relu = nn.ReLU()
#     self.fc2 = nn.Linear(128, 64)
#     self.relu = nn.ReLU()
#     self.fc3 = nn.Linear(64, 6)
#
#   def forward(self, res2):
#     # Flatten the tensor to match the input shape of the fully connected layer
#     res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]
#
#     # Pass through fully connected layers
#     # res2 = self.fc1(res2)
#     # res2 = self.relu(res2)
#     res2 = self.fc2(res2)
#     res2 = self.relu(res2)
#     out = self.fc3(res2)
#
#     return out

class Sub_Net_bxl_816(nn.Module):  # luma bt depth and direction prediction
  def __init__(self):
    super(Sub_Net_bxl_816, self).__init__()

    # Attention modules
    # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
    # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))

    # Final fully connected layers for output matching subnet3
    self.fc1 = nn.Linear(512, 256)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 64)
    self.relu = nn.ReLU()
    self.fc3 = nn.Linear(64, 6)

    self.qp_fc = nn.Linear(1,64)

  def forward(self, res2, qp):
    # Flatten the tensor to match the input shape of the fully connected layer
    res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]

    # Pass through fully connected layers
    # res2 = self.fc1(res2)
    # res2 = self.relu(res2)
    res2 = self.fc2(res2)

    qp = qp.view(-1, 1).float()
    qp_feat = self.qp_fc(qp)

    res2 = res2 + qp_feat

    res2 = self.relu(res2)
    out = self.fc3(res2)

    return out

class Sub_Net_bxl_1632(nn.Module):  # luma bt depth and direction prediction
    def __init__(self):
      super(Sub_Net_bxl_1632, self).__init__()

      # Attention modules
      # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
      # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))

      # Final fully connected layers for output matching subnet3
      self.fc1 = nn.Linear(512, 256)
      self.relu = nn.ReLU()
      self.fc2 = nn.Linear(256, 64)
      self.relu = nn.ReLU()
      self.fc3 = nn.Linear(64, 6)

      self.qp_fc = nn.Linear(1, 64)

    def forward(self, res2, qp):
      # Flatten the tensor to match the input shape of the fully connected layer
      res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]

      # Pass through fully connected layers
      res2 = self.fc1(res2)
      res2 = self.relu(res2)
      res2 = self.fc2(res2)

      qp = qp.view(-1, 1).float()
      qp_feat = self.qp_fc(qp)

      res2 = res2 + qp_feat

      res2 = self.relu(res2)
      out = self.fc3(res2)

      return out


class Sub_Net_bxl_6464(nn.Module):  # luma bt depth and direction prediction
  def __init__(self):
    super(Sub_Net_bxl_6464, self).__init__()

    # Attention modules
    # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
    # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))

    # Final fully connected layers for output matching subnet3
    self.fc1 = nn.Linear(1024, 256)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(256, 64)
    self.relu = nn.ReLU()
    self.fc3 = nn.Linear(64, 6)

  def forward(self, res2):
    # Flatten the tensor to match the input shape of the fully connected layer
    res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]

    # Pass through fully connected layers
    res2 = self.fc1(res2)
    res2 = self.relu(res2)
    res2 = self.fc2(res2)
    res2 = self.relu(res2)
    out = self.fc3(res2)

    return out

class Sub_Net_bxl_3232(nn.Module):  # luma bt depth and direction prediction
  def __init__(self):
    super(Sub_Net_bxl_3232, self).__init__()

    # Attention modules
    # self.atten_module_1 = nn.Parameter(torch.ones(atten_input, 16))
    # self.atten_module_2 = nn.Parameter(torch.ones(atten_input, 64))

    # Final fully connected layers for output matching subnet3
    self.fc1 = nn.Linear(1024, 256)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(256, 64)
    self.relu = nn.ReLU()
    self.fc3 = nn.Linear(64, 6)

    self.qp_fc = nn.Linear(1, 64)

  def forward(self, res2, qp):
    # Flatten the tensor to match the input shape of the fully connected layer
    res2 = res2.view(res2.size(0), -1)  # [batch_size, num_features]

    # Pass through fully connected layers
    res2 = self.fc1(res2)
    res2 = self.relu(res2)
    res2 = self.fc2(res2)

    qp = qp.view(-1, 1).float()
    qp_feat = self.qp_fc(qp)

    res2 = res2 + qp_feat

    res2 = self.relu(res2)
    out = self.fc3(res2)

    return out
  # def forward(self, x0_0):  # 1x64x64, 1x64x64
  #   # x0_2 = F.relu(self.conv_q1(x0_0))  # 8*64*64
  #   # x1_1 = F.interpolate(F.max_pool2d(x0_2, 4), scale_factor=4)
  #   # x1_2 = F.interpolate(F.max_pool2d(x0_2, 8), scale_factor=8)
  #   # x1_3 = F.interpolate(F.max_pool2d(x0_2, 16), scale_factor=16)
  #   x1_1 = F.relu(self.conv_b1_1(self.padding_luma1(x0_0)))  # 16*64*64
  #   x1_2 = F.relu(self.conv_b1_2(self.padding_r(x0_0)))  # 8*64*64
  #   x1_3 = F.relu(self.conv_b1_3(self.padding_b(x0_0)))  # 8*64*64
  #   x2 = torch.cat([x1_1, x1_2, x1_3], 1)  # 32*64*64
  #   x2_1 = F.max_pool2d(self.trunk_M0(x2), 2)  # 32*32*32
  #
  #
  #   x4 = self.padding_luma(x2_1)  # 32x34x34
  #
  #
  #   x5 = F.max_pool2d((self.trunk_M1(x4)), 2)  # 32*17*17 M2 out
  #   x6 = self.trunk_M2(x5)  # 16*17*17
  #   x7 = self.trunk_B1(x6)  # 8*17*17
  #   out = self.conv_B1(x7)  # 2*17*17
  #   return out


class Edge_Net_withoutedge(nn.Module):  # luma bt depth and direction prediction
    def __init__(self):
        super(Edge_Net_withoutedge, self).__init__()
        self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
        self.padding_rb = nn.ZeroPad2d((0, 8, 0, 8))
        self.padding_r = nn.ZeroPad2d((4, 4, 2, 2))
        self.padding_b = nn.ZeroPad2d((2, 2, 4, 4))
        self.padding_luma = nn.ZeroPad2d((1, 1, 1, 1))
        self.padding_luma1 = nn.ZeroPad2d((4, 4, 4, 4))
        self.conv_b1_1 = nn.Conv2d(1, 16, kernel_size=(9, 9), padding=0, stride=1)
        self.conv_b1_2 = nn.Conv2d(1, 8, kernel_size=(5, 9), padding=0, stride=1)
        self.conv_b1_3 = nn.Conv2d(1, 8, kernel_size=(9, 5), padding=0, stride=1)
        self.conv_q1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
        self.resblock_q1 = ResidualBlock(16, 16, kernel_size=5, padding=2)
        self.resblock_q2 = ResidualBlock(32, 16, kernel_size=5, padding=2)

        # M-Main, B-Branch, A-Attention
        self.trunk_M0 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1))
        self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1),
                                      ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
                                      ResidualBlock(32, 32, 3, 1),ResidualBlock(32, 32, 3, 1))
        self.trunk_M2 = nn.Sequential(ResidualBlock(32, 32, 5, 2), ResidualBlock(32, 32, 5, 2),
                                      ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
                                      ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 16, 3, 1))
        self.trunk_B1 = nn.Sequential(ResidualBlock(16, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
        self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
        self.trunk_Att1 = nn.Sequential(ResidualBlock(1, 16, 3, 1), ResidualBlock(16, 32, 3, 1))

    # x0_0 原始像素
    # x0_1 canny点图 1 2
    def forward(self, x0_0):  # 1x64x64, 1x64x64
        # x0_2 = F.relu(self.conv_q1(x0_0))  # 8*64*64
        # x1_1 = F.interpolate(F.max_pool2d(x0_2, 4), scale_factor=4)
        # x1_2 = F.interpolate(F.max_pool2d(x0_2, 8), scale_factor=8)
        # x1_3 = F.interpolate(F.max_pool2d(x0_2, 16), scale_factor=16)
        x1_1 = F.relu(self.conv_b1_1(self.padding_luma1(x0_0)))  # 16*64*64
        x1_2 = F.relu(self.conv_b1_2(self.padding_r(x0_0)))  # 8*64*64
        x1_3 = F.relu(self.conv_b1_3(self.padding_b(x0_0)))  # 8*64*64
        x2 = torch.cat([x1_1, x1_2, x1_3], 1)  # 32*64*64
        x2_1 = F.max_pool2d(self.trunk_M0(x2),2) # 32*32*32

        # x2_att = F.max_pool2d(self.trunk_Att1(x0_1),2)  # 32x32x32
        # x3 = x2_1 * x2_att  # 32x32x32
        x4 = self.padding_luma(x2_1) # 32x34x34

        # x4 = F.max_pool2d(self.trunk_M1(x3), 2)  # 64*34*34 M1 out
        x5 = F.max_pool2d((self.trunk_M1(x4)), 2)  # 32*17*17 M2 out
        x6 = self.trunk_M2(x5)  # 16*17*17
        x7 = self.trunk_B1(x6)  # 8*17*17
        out = self.conv_B1(x7)  # 2*17*17
        return out



# class Edge_Net(nn.Module):  # luma bt depth and direction prediction
#     def __init__(self):
#         super(Edge_Net, self).__init__()
#         self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
#         self.padding_rb = nn.ZeroPad2d((0, 4, 0, 4))
#         self.padding_r = nn.ZeroPad2d((0, 4, 0, 0))
#         self.padding_b = nn.ZeroPad2d((0, 0, 0, 4))
#         self.padding_luma = nn.ZeroPad2d(( 2, 2, 2, 2))
#         self.padding_luma1 = nn.ZeroPad2d((1, 1, 1, 1))
#         self.conv_q1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
#         self.resblock_q1 = ResidualBlock(16, 16, kernel_size=5, padding=2)
#         self.resblock_q2 = ResidualBlock(32, 16, kernel_size=5, padding=2)
#
#         # M-Main, B-Branch, A-Attention
#         self.trunk_M0 = nn.Sequential(ResidualBlock(64, 64, 5, 2), ResidualBlock(64, 32, 3, 1),
#                                       ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 32, 3, 1))
#         self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
#                                       ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1))
#         self.trunk_M2 = nn.Sequential(ResidualBlock(64, 64, 5, 2), ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
#                                       ResidualBlock(32, 16, 3, 1))
#         self.trunk_B1 = nn.Sequential(ResidualBlock(16, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
#         self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
#         self.trunk_Att1 = nn.Sequential(ResidualBlock(1, 16, 3, 1), ResidualBlock(16, 32, 3, 1))
#
#     def forward(self, x0_0, x0_1):  # 1x64x64, 1x64x64
#
#         x0_2 = self.padding_luma(x0_0) # 1x68x68
#         x0_3 = F.relu(self.conv_q1(x0_2))  # 16*68*68
#         x1_0 = F.relu(self.resblock_q1(x0_3)) # 16*68*68
#         x1_1 = F.interpolate(F.max_pool2d(x0_3, 2), scale_factor=2)
#         x1_2 = F.interpolate(F.max_pool2d(x0_3, 4), scale_factor=4)
#         x2 = torch.cat([x0_3, x1_0, x1_1, x1_2], 1)  # 64*68*68
#         x2_0 = F.max_pool2d(self.trunk_M0(x2),2) # 32x34x34
#
#         x2_1 = self.padding_luma1(F.interpolate(x0_1, scale_factor=2))  # 1*34*34
#         x2_att = self.trunk_Att1(x2_1) # 32x34x34
#         x3 = x2_0 * x2_att # 32x34x34
#
#         # x4 = F.max_pool2d(self.trunk_M1(x3), 2)  # 64*34*34 M1 out
#         x5 = F.max_pool2d(self.trunk_M2(self.trunk_M1(x3)), 2)  # 32*17*17 M2 out
#         x6 = self.trunk_B1(x5)  # 8*17*17
#         out = self.conv_B1(x6)  # 2*17*17
#         return out


# class Edge_Net(nn.Module):  # luma bt depth and direction prediction
#     def __init__(self):
#         super(Edge_Net, self).__init__()
#         self.padding_lu = nn.ZeroPad2d((4, 0, 4, 0))
#         self.padding_rb = nn.ZeroPad2d((0, 4, 0, 4))
#         self.padding_r = nn.ZeroPad2d((0, 4, 0, 0))
#         self.padding_b = nn.ZeroPad2d((0, 0, 0, 4))
#         self.padding_luma = nn.ZeroPad2d(( 2, 2, 2, 2))
#         # M-Main, B-Branch, A-Attention
#         self.trunk_M1 = nn.Sequential(ResidualBlock(32, 64, 5, 2), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1),
#                                       ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1), ResidualBlock(64, 64, 3, 1))
#         self.trunk_M2 = nn.Sequential(ResidualBlock(64, 32, 3, 1), ResidualBlock(32, 32, 3, 1), ResidualBlock(32, 32, 3, 1),
#                                       ResidualBlock(32, 32, 3, 1))
#         self.trunk_B0 = nn.Sequential(ResidualBlock(33, 16, 3, 1))
#         self.trunk_B1 = nn.Sequential(ResidualBlock(32, 16, 3, 1), ResidualBlock(16, 8, 3, 1))
#         self.conv_B1 = nn.Conv2d(8, 2, kernel_size=3, padding=1, stride=1)
#         self.trunk_Att1 = nn.Sequential(ResidualBlock(1, 8, 3, 1), ResidualBlock(8, 16, 3, 1))
#
#     def forward(self, x0_1, x0_2, x1):  # 32x64x64, 1x64x64, 1x16x16
#         x0_3 = torch.cat([x0_1,x0_2],1) #33x64x64
#         x0_4 = self.trunk_B0(self.padding_luma(F.max_pool2d(x0_3)))  # 16x34x34
#         x1_1 = self.padding_luma(F.interpolate(x1, scale_factor=2))  # 1*34*34
#         x1_att = self.trunk_Att1(x1_1)
#         x2 = x * x1_att # 16x34x34
#
#         x3 = F.max_pool2d(self.trunk_M1(x2), 2)  # 64*34*34 M1 out
#         x4 = F.max_pool2d(self.trunk_M2(x3), 2)  # 32*17*17 M2 out
#         x5 = self.trunk_B1(x4)  # 8*17*17
#         out = self.conv_B1(x5)  # 2*17*17
#         return out


def PywtTransfrom(x):
    ca, (ch, cv, cd) = pywt.dwt2(x, 'haar')
    return ca, ch, cv, cd


if __name__ == '__main__':
    model_MT = MT_Net()
    model_Edge = Edge_Net()

    # model_BD = BD_Net(EPSABlock)
    # model_QBD = QBD_Net()
    # print(model)
    # print("params:",sum(param.numel() for param in model.parameters()))
    a = time.time()
    x = torch.randn(1, 1, 64, 64)
    # x1 = torch.randn(1, 32, 64, 64)
    # canny_factor = cv2.Canny(x, 37*2, 37*4)
    canny_mat = torch.randn(1, 1, 64, 64)
    # height = 1344 // 64
    # width = 1920 // 64
    # canny_mat =
    # for i in range(height):
    #     for j in range(width):
    #         if canny_factor[i, j] == 255:
    #             canny_factor[i, j // 4] += 1
    # x2 = torch.randn(1, 1, 16, 16)
    # out = model_MT(x)  # 32x32x32
    # out = model_Edge(x1,x2)
    out = model_Edge(x, canny_mat)

    print(time.time() - a)
    print(out.shape)
    # macs,params = get_model_complexity_info(model,shape,as_strings=False,print_per_layer_stat=False,verbose=True)
    # print('MACs:  ',  macs/64/64)
    # print('Params: ' , params)
