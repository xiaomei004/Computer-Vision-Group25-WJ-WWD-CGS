"""CIFAR-10 数据加载与预处理。"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# CIFAR-10 各通道均值和标准差（训练集统计值）
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# ImageNet 均值和标准差（ResNet18 预训练权重使用的归一化）
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def build_dataloaders(model: str, batch_size: int = 128, num_workers: int = 2):
    """构建 CIFAR-10 的训练集与测试集 DataLoader。

    Args:
        model: "simple"（32x32，CIFAR 归一化）或 "resnet18"（224x224，ImageNet 归一化）。
        batch_size: 批大小。
        num_workers: 数据加载子进程数。
    """
    if model == "resnet18":
        input_size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD
    else:
        input_size, mean, std = 32, CIFAR_MEAN, CIFAR_STD

    if model == "resnet18":
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
