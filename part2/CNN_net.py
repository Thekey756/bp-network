import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time





class CNNNet(nn.Module):

    def __init__(self):
        super(CNNNet, self).__init__()        # 父类初始化
        # 输入图像是单通道，conv1 kenrnel size=5*5，输出通道 6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        # conv2 kernel size=5*5, 输出通道 16
        self.conv2 = nn.Conv2d(16, 32, 3)
        # 池化层
        self.pool = nn.MaxPool2d(2)
        # 激活层
        # self.relu = nn.ReLU()
        # 全连接层
        self.fc1 = nn.Linear(32*5*5, 120)
        # self.fc2 = nn.Linear(84,12)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 12)

    def forward(self, x):
        # max-pooling 采用一个 (2,2) 的滑动窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 核(kernel)大小是方形的话，可仅定义一个数字，如 (2,2) 用 2 即可
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print("+++++++++++++++++++++++++++++++++++++++++")
        # print(x.size())
        # print("x.size = ".format(x.size(0)))
        # print("+++++++++++++++++++++++++++++++++++++++++")
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 除了 batch 维度外的所有维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features