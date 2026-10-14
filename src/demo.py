"""演示脚本：加载训练好的模型，现场做实时预测。

用法：
    # 对测试集前 N 张图片做预测，并保存一张对比图
    python src/demo.py --model resnet18 --weights results/resnet18/best.pth --n 6

    # 对一张自定义图片（如手机拍的照片）做预测，输出各类别概率
    python src/demo.py --model resnet18 --weights results/resnet18/best.pth --image 你的图片.jpg
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

from data import DATA_DIR, CIFAR10_CLASSES, CIFAR_MEAN, CIFAR_STD, IMAGENET_MEAN, IMAGENET_STD
from model import SimpleCNN, get_resnet18

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["simple", "resnet18"], default="resnet18")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--image", type=str, default=None, help="自定义图片路径（不填则用测试集样本）")
    p.add_argument("--n", type=int, default=6, help="展示的测试集样本数")
    return p.parse_args()


def build_model(name):
    if name == "resnet18":
        return get_resnet18(num_classes=10, pretrained=False)
    return SimpleCNN(num_classes=10)


def make_transform(name):
    if name == "resnet18":
        return transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])


def display_stats(name):
    if name == "resnet18":
        return IMAGENET_MEAN, IMAGENET_STD
    return CIFAR_MEAN, CIFAR_STD


@torch.no_grad()
def predict_probs(model, x, device):
    model.eval()
    out = model(x.unsqueeze(0).to(device))
    return torch.softmax(out, dim=1)[0].cpu()


def topk(probs, k=5):
    v, i = probs.topk(k)
    return list(zip(i.tolist(), v.tolist()))


def predict_custom_image(model, tf, image_path, device):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    probs = predict_probs(model, tf(img), device)
    print(f"\n图片：{image_path}")
    print("模型预测（Top-5，概率从高到低）：")
    for idx, p in topk(probs, 5):
        bar = "#" * int(p * 50)
        print(f"  {CIFAR10_CLASSES[idx]:<12s} {p * 100:5.1f}%  {bar}")
    print(f"\n最终判定：{CIFAR10_CLASSES[int(probs.argmax())]}（置信度 {probs.max() * 100:.1f}%）")


def predict_test_samples(model, name, tf, device, n):
    raw = datasets.CIFAR10(DATA_DIR, train=False, download=False)
    imgs = [raw[i][0] for i in range(n)]
    labels = [raw[i][1] for i in range(n)]
    x = torch.stack([tf(img) for img in imgs])
    labels_t = torch.tensor(labels)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x.to(device)), dim=1).cpu()
    preds = probs.argmax(1)

    for i in range(n):
        top3 = topk(probs[i], 3)
        s = "  ".join(f"{CIFAR10_CLASSES[j]}({p * 100:.0f}%)" for j, p in top3)
        flag = "对" if preds[i] == labels_t[i] else "错"
        print(f"[{flag}] 真实={CIFAR10_CLASSES[labels[i]]:<12s} "
              f"预测={CIFAR10_CLASSES[preds[i]]:<12s} Top3: {s}")

    mean, std = display_stats(name)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if n > 1 else [axes]
    for i in range(n):
        show = (x[i] * std_t + mean_t).clamp(0, 1)  # 去归一化以便显示
        axes[i].imshow(show.permute(1, 2, 0))
        ok = preds[i] == labels_t[i]
        axes[i].set_title(f"预测:{CIFAR10_CLASSES[preds[i]]}\n真实:{CIFAR10_CLASSES[labels[i]]}",
                          fontsize=10, color="green" if ok else "red")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(f"{name} 现场预测演示（绿=正确，红=错误）", fontsize=14)
    fig.tight_layout()
    out = RESULTS_DIR / "demo_prediction.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n对比图已保存到 {out}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    tf = make_transform(args.model)
    print(f"[demo] 模型 {args.model} 已加载，设备 {device}\n")

    if args.image:
        predict_custom_image(model, tf, args.image, device)
    else:
        predict_test_samples(model, args.model, tf, device, args.n)


if __name__ == "__main__":
    main()
