import torch
from torch import nn  # 用于构建模型和损失函数
from torch.utils.data import DataLoader  # 用于构建数据加载器
from torchvision import datasets, transforms  # 用于读入和变换数据集

data = datasets.CIFAR10(root="data", train="True", download="True")