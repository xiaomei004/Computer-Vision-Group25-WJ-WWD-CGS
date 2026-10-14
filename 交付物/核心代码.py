######################################################################
# 文件：src/model.py
######################################################################

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



######################################################################
# 文件：src/data.py
######################################################################

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



######################################################################
# 文件：src/train.py
######################################################################

"""模型训练脚本。

用法示例：
    python src/train.py --model simple --epochs 20 --lr 1e-3
    python src/train.py --model resnet18 --epochs 8 --lr 1e-3 --freeze
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import build_dataloaders
from model import SimpleCNN, get_resnet18

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simple", "resnet18"], default="simple")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--freeze", action="store_true",
                   help="冻结 ResNet18 骨干，仅训练分类头")
    p.add_argument("--out", type=str, default=None, help="输出子目录名")
    return p.parse_args()


def build_model(name, freeze):
    if name == "resnet18":
        model = get_resnet18(num_classes=10, pretrained=True)
        if freeze:
            for p in model.parameters():
                p.requires_grad = False
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
    else:
        model = SimpleCNN(num_classes=10)
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] model={args.model} device={device}")

    train_loader, test_loader = build_dataloaders(args.model, args.batch_size)
    model = build_model(args.model, args.freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = RESULTS_DIR / (args.out or args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_path = out_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"epoch {epoch:02d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, out_dir / "training_curves.png")
    print(f"[train] 完成，最佳测试准确率 = {best_acc:.4f}，权重已保存到 {best_path}")


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.set_title("Loss")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"], label="test")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()
    ax2.set_title("Accuracy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()



######################################################################
# 文件：src/evaluate.py
######################################################################

"""模型评估脚本：计算测试集准确率、绘制混淆矩阵。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import torch

from data import build_dataloaders, CIFAR10_CLASSES
from model import SimpleCNN, get_resnet18

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simple", "resnet18"], default="simple")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def build_model(name):
    if name == "resnet18":
        return get_resnet18(num_classes=10, pretrained=False)
    return SimpleCNN(num_classes=10)


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds), torch.cat(all_labels)


def confusion_matrix(preds, labels, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds.tolist(), labels.tolist()):
        cm[l, p] += 1
    return cm


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵 (Confusion Matrix)")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = build_dataloaders(args.model, batch_size=128, num_workers=2)

    model = build_model(args.model).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)

    preds, labels = predict_all(model, test_loader, device)
    acc = (preds == labels).float().mean().item()
    cm = confusion_matrix(preds, labels)

    out_dir = RESULTS_DIR / (args.out or args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_class_acc = {CIFAR10_CLASSES[i]: float(cm[i, i] / cm[i].sum()) for i in range(10)}

    print(f"[eval] 测试集准确率 = {acc:.4f}")
    for name, a in per_class_acc.items():
        print(f"  {name:<10s}: {a:.4f}")

    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"test_acc": acc, "per_class_acc": per_class_acc}, ensure_ascii=False, indent=2))
    plot_confusion_matrix(cm, CIFAR10_CLASSES, out_dir / "confusion_matrix.png")
    print(f"[eval] 混淆矩阵已保存到 {out_dir}")


if __name__ == "__main__":
    main()


