"""一键运行：训练 SimpleCNN 与 ResNet18，并评估。

用法：
    python src/run_all.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n" + "=" * 60)
    print("$", " ".join(cmd))
    print("=" * 60)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    py = sys.executable
    # 1) 训练基础 CNN（从零训练）
    run([py, "src/train.py", "--model", "simple", "--epochs", "20", "--lr", "1e-3"])
    # 2) 训练 ResNet18 迁移学习（加载 ImageNet 预训练，全量微调）
    run([py, "src/train.py", "--model", "resnet18", "--epochs", "15", "--lr", "1e-3", "--batch-size", "64"])
    # 3) 评估两个模型
    run([py, "src/evaluate.py", "--model", "simple", "--weights", "results/simple/best.pth"])
    run([py, "src/evaluate.py", "--model", "resnet18", "--weights", "results/resnet18/best.pth"])
    print("\n全部完成！结果保存在 results/ 目录。")


if __name__ == "__main__":
    main()
