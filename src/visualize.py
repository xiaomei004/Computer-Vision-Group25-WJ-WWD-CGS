"""生成测试集样本的预测结果图（预测标签 vs 真实标签）。

用法：
    python src/visualize.py --model simple   --weights results/simple/best.pth
    python src/visualize.py --model resnet18 --weights results/resnet18/best.pth
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
from torchvision import datasets, transforms

from data import build_dataloaders, CIFAR10_CLASSES, DATA_DIR, CIFAR_MEAN, CIFAR_STD, IMAGENET_MEAN, IMAGENET_STD
from model import SimpleCNN, get_resnet18

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simple", "resnet18"], default="simple")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--n", type=int, default=24, help="展示的样本数")
    return p.parse_args()


def build_model(name):
    if name == "resnet18":
        return get_resnet18(num_classes=10, pretrained=False)
    return SimpleCNN(num_classes=10)


def norm_transform(model):
    if model == "resnet18":
        return transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 原始 PIL 数据集，用于展示；推理时再套归一化
    raw_set = datasets.CIFAR10(DATA_DIR, train=False, download=False)
    norm_tf = norm_transform(args.model)

    indices = list(range(args.n))
    raw_imgs = torch.stack([transforms.ToTensor()(raw_set[i][0]) for i in indices])
    labels = torch.tensor([raw_set[i][1] for i in indices])
    norm_imgs = torch.stack([norm_tf(raw_set[i][0]) for i in indices])

    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = model(norm_imgs.to(device)).argmax(1).cpu()

    # 去归一化以便显示
    mean = torch.tensor(IMAGENET_MEAN if args.model == "resnet18" else CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD if args.model == "resnet18" else CIFAR_STD).view(1, 3, 1, 1)
    show = (norm_imgs * std + mean).clamp(0, 1)

    cols = 8
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    axes = axes.flatten()
    for i in range(args.n):
        ax = axes[i]
        ax.imshow(show[i].permute(1, 2, 0))
        correct = preds[i] == labels[i]
        ax.set_title(f"预测:{CIFAR10_CLASSES[preds[i]]}\n真实:{CIFAR10_CLASSES[labels[i]]}",
                     fontsize=7, color="green" if correct else "red")
        ax.axis("off")
    for i in range(args.n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"{args.model} 样本预测（绿=正确，红=错误）", fontsize=13)
    fig.tight_layout()
    out_dir = RESULTS_DIR / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "prediction_samples.png", dpi=150)
    plt.close(fig)
    print(f"[vis] 预测图已保存到 {out_dir / 'prediction_samples.png'}")


if __name__ == "__main__":
    main()
