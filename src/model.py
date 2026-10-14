"""模型定义：SimpleCNN 与 ResNet18 迁移学习模型。

本文件包含两个用于 CIFAR-10 图像分类的模型：
- SimpleCNN：一个从零搭建的基础卷积神经网络。
- ResNet18：基于 ImageNet 预训练的迁移学习模型。
"""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SimpleCNN(nn.Module):
    """基础卷积神经网络，输入为 3x32x32 的 CIFAR-10 图像。

    结构：三个 (卷积 -> 批归一化 -> ReLU -> 最大池化) 块，再接两层全连接。
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 64x8x8

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 128x4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_resnet18(num_classes: int = 10, pretrained: bool = True):
    """ResNet18 迁移学习模型。

    使用 ImageNet 预训练权重初始化，仅替换最后一层全连接层以适应 CIFAR-10 的 10 类。
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
