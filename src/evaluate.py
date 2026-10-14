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
