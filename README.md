# 计算机视觉大作业 · CNN 图像分类

> 课程：大四上 2026.10 计算机视觉
> 课题：③ 使用 CNN 进行图像分类（难度系数 1.0）
> 组号：第 25 组 · 成员：WJ、WWD、CGS

基于 **CIFAR-10** 数据集实现并对比两种卷积神经网络分类方案：
- **SimpleCNN**：从零搭建的轻量级卷积网络（基线）
- **ResNet18 迁移学习**：ImageNet 预训练 + 全量微调

最终测试集准确率：SimpleCNN **80.31%**，ResNet18 **95.18%**。

---

## 成员与分工

| 成员 | 职责 |
|---|---|
| WJ | 数据预处理与环境搭建（`src/data.py`、`requirements.txt`、`README.md`） |
| WWD | 模型设计与训练（`src/model.py`、`src/train.py`） |
| CGS | 实验评估与报告撰写（`src/evaluate.py`、`报告/`） |

详见 `报告/成员名单与职责分配.md`。

---

## 环境要求

- Python 3.11
- （推荐）NVIDIA GPU，CUDA 驱动 ≥ 12.6
- PyTorch 2.14.1

## 安装

```bash
# GPU 版（CUDA 13.0）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 无 GPU 用 CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 其余依赖
pip install -r requirements.txt
```

## 数据集下载（国内加速）

CIFAR-10 默认从 `cs.toronto.edu` 下载，国内访问极慢。若自动下载卡住，可手动下载到 `data/` 目录（torchvision 会自动识别、跳过下载）：

```bash
curl -L -o data/cifar-10-python.tar.gz "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
```

文件 MD5 应为 `c58f30108f718f92721af3b95e74349a`。

---

## 一键复现

```bash
python src/run_all.py
```

该脚本依次执行：训练 SimpleCNN（20 epoch）→ 训练 ResNet18 迁移学习（15 epoch）→ 评估两个模型并输出准确率与混淆矩阵。

也可以分步执行：

```bash
# 训练
python src/train.py --model simple   --epochs 20 --lr 1e-3
python src/train.py --model resnet18 --epochs 15 --lr 1e-3 --batch-size 64

# 评估
python src/evaluate.py --model simple   --weights results/simple/best.pth
python src/evaluate.py --model resnet18 --weights results/resnet18/best.pth
```

所有结果输出到 `results/` 目录。

---

## 实验结果

| 模型 | 训练方式 | 测试集准确率 |
|---|---|---|
| SimpleCNN | 从零训练 | 80.31% |
| ResNet18 | 迁移学习 | 95.18% |

- 训练曲线：`results/simple/training_curves.png`、`results/resnet18/training_curves.png`
- 混淆矩阵：`results/simple/confusion_matrix.png`、`results/resnet18/confusion_matrix.png`

---

## 答辩现场演示

加载训练好的权重，即可现场演示（无需重新训练）：

```bash
# ① 实时预测：对测试集样本逐张预测并生成对比图
python src/demo.py --model resnet18 --weights results/resnet18/best.pth --n 6

# ② 评估：现场打印准确率 + 各类别准确率 + 混淆矩阵
python src/evaluate.py --model resnet18 --weights results/resnet18/best.pth

# ③ 自定义图片预测（输出 Top-5 概率）
python src/demo.py --model resnet18 --weights results/resnet18/best.pth --image 你的图片.jpg
```

> 注：`best.pth` 权重体积较大且已被 gitignore。队友克隆仓库后需先运行 `python src/run_all.py` 训练生成权重，再执行上述演示命令。

---

## 目录结构

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── 学习笔记.md          # 概念梳理与答辩速通
├── src/
│   ├── model.py        # SimpleCNN 与 ResNet18 模型定义
│   ├── data.py         # CIFAR-10 数据加载与增强
│   ├── train.py        # 训练脚本
│   ├── evaluate.py     # 评估脚本（准确率/混淆矩阵）
│   ├── visualize.py    # 预测样本可视化
│   ├── demo.py         # 答辩演示：实时预测
│   └── run_all.py      # 一键复现入口
├── data/               # CIFAR-10（自动/手动下载，已 gitignore）
├── results/            # 权重、曲线、混淆矩阵（*.pth 已 gitignore）
├── tools/              # 交付物生成脚本
├── 报告/               # 汇总报告、个人报告、调优记录、PPT 文案
└── 交付物/             # Word 报告、Excel 分工表、核心代码
```

---

## 报告与答辩材料

- 汇总报告：`报告/汇总报告.md`
- 个人报告：`报告/个人报告_WJ.md`、`报告/个人报告_WWD.md`、`报告/个人报告_CGS.md`
- 成员名单与职责分配：`报告/成员名单与职责分配.md`
- 实验调优过程记录：`报告/实验调优过程记录.md`
- 答辩 PPT 文案：`报告/答辩PPT文案.md`
- 学习笔记（答辩速通）：`学习笔记.md`
- 交付物：`交付物/`（Word 报告、Excel 分工表、核心代码）
