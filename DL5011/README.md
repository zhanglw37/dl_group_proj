# Frame Prediction Using InstructPix2Pix

DL5011课程大作业 - 基于InstructPix2Pix的视频帧预测

## 项目概述

本项目微调 [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix) 模型，用于预测 Something-Something V2 数据集中人-物交互视频的未来帧。给定视频的初始帧（第0帧）和动作的文本描述，模型能够预测21帧后的场景。

### 任务定义

- **输入**: 初始帧（第0帧） + 文本描述
- **输出**: 预测的第21帧
- **三个任务类别**:
  - `move_object`: 移动物体（左右推动）
  - `drop_object`: 放下/掉落物体
  - `cover_object`: 覆盖物体

### 数据集统计

- **训练集**: 284个样本
  - move_object: 95
  - drop_object: 93
  - cover_object: 96
- **验证集**: 72个样本（每个任务24个）
- **图像分辨率**: 256×256

## 环境配置

### 1. 创建Conda环境

```bash
cd instruct-pix2pix
conda env create -f environment.yaml
conda activate ip2p
```

### 2. 安装额外依赖

```bash
pip install scikit-image
pip install diffusers
```

### 3. 下载预训练模型

预训练的InstructPix2Pix模型已下载至 `checkpoint/` 目录。

## 数据准备

### 1. 下载 Something-Something V2 数据集

将视频文件放置在: `dataset/20bn-something-something-v2/`

### 2. 处理数据集

本项目支持两种预测策略：
- **frame0_21**: 第0帧预测第21帧（长期预测，跨度约0.7秒）
- **frame20_21**: 第20帧预测第21帧（短期预测，跨度约0.03秒）

```bash
# 策略1: 第0帧 -> 第21帧（默认）
python prepare_dataset.py \
    --max_samples 120 \
    --output_dir processed_data \
    --size 256 \
    --strategy frame0_21

# 策略2: 第20帧 -> 第21帧
python prepare_dataset.py \
    --max_samples 120 \
    --output_dir processed_data_20_21 \
    --size 256 \
    --strategy frame20_21
```

参数说明:
- `--max_samples`: 每个任务类别的最大样本数
- `--output_dir`: 输出目录
- `--size`: 图像尺寸（正方形）
- `--strategy`: 预测策略（frame0_21 或 frame20_21）

处理后的数据结构:
```
processed_data/
├── move_object/
│   ├── {video_id}_input.png
│   └── {video_id}_target.png
├── drop_object/
├── cover_object/
└── dataset_info.json
```

## 训练

### 基本训练命令

```bash
# 策略1: 第0帧 -> 第21帧
python train.py \
    --config configs/train_frame_prediction.yaml \
    --gpus 0 \
    --seed 42

# 策略2: 第20帧 -> 第21帧
python train.py \
    --config configs/train_frame_prediction_20_21.yaml \
    --gpus 0 \
    --seed 42
```

### 参数说明

- `--config`: 配置文件路径
- `--gpus`: 使用的GPU ID（逗号分隔，如 "0,1"）
- `--seed`: 随机种子
- `--resume`: 从检查点继续训练（可选）
- `--logdir`: 日志目录（默认: `logs/`）

### 配置文件

主要参数在 `configs/train_frame_prediction.yaml`:

```yaml
model:
  base_learning_rate: 5.0e-05  # 学习率
  params:
    ckpt_path: checkpoint/instruct-pix2pix-00-22000.ckpt  # 预训练模型

data:
  params:
    batch_size: 2  # 批量大小
    num_workers: 4

lightning:
  trainer:
    max_epochs: 100
    precision: 16  # 半精度训练
    accumulate_grad_batches: 2  # 梯度累积
```

### 训练监控

训练日志保存在 `logs/frame_prediction_{timestamp}/`:
- TensorBoard日志: `tensorboard/`
- 模型检查点: `checkpoints/`

查看训练进度:
```bash
# 启动TensorBoard
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006

# 或指定具体的训练日志目录
tensorboard --logdir logs/frame_prediction_20251120_204144/tensorboard --host 0.0.0.0 --port 6006
```

然后在浏览器中打开 `http://localhost:6006` 查看训练曲线。

## 评估

### 运行评估

使用 `evaluate_lightning.py` 从PyTorch Lightning checkpoint加载模型并计算指标：

```bash
python evaluate_lightning.py \
    --config configs/train_frame_prediction.yaml \
    --checkpoint logs/frame_prediction_XXXXXXXX_XXXXXX/checkpoints/frame-pred-epoch=XX-val/loss_simple_ema=X.XXXX.ckpt \
    --data_root processed_data \
    --output_dir evaluation_results \
    --split val \
    --steps 50 \
    --device cuda
```

### 参数说明

- `--config`: 训练配置文件路径
- `--checkpoint`: Lightning checkpoint文件路径
- `--data_root`: 处理后的数据目录
- `--output_dir`: 评估结果输出目录
- `--split`: 数据集划分（train/val）
- `--steps`: DDIM采样步数（默认50）
- `--device`: 计算设备（cuda/cpu）

### 评估示例

```bash
# 评估Epoch 1的checkpoint
python evaluate_lightning.py \
    --config configs/train_frame_prediction.yaml \
    --checkpoint logs/frame_prediction_20251120_204144/checkpoints/frame-pred-epoch=01-val/loss_simple_ema=0.1371.ckpt \
    --data_root processed_data \
    --output_dir evaluation_results_epoch01 \
    --split val \
    --steps 50 \
    --device cuda
```

### 评估指标

- **SSIM** (Structural Similarity Index): 结构相似度，范围 [0, 1]，越高越好
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比，单位 dB，越高越好

### 评估结果

评估结果保存在输出目录中：
- `evaluation_results.json`: 数值结果（总体和各任务的SSIM/PSNR）
- `{task}/{video_id}_ssim{val}_psnr{val}.png`: 可视化结果（输入|预测|真值）

示例输出：
```
总体:
  SSIM: 0.3076 ± 0.1646
  PSNR: 11.51 ± 2.61

各任务:
  move_object (n=24):
    SSIM: 0.3497 ± 0.1790
    PSNR: 12.34 ± 2.53
  drop_object (n=24):
    SSIM: 0.3044 ± 0.1496
    PSNR: 11.34 ± 2.44
  cover_object (n=24):
    SSIM: 0.2688 ± 0.1536
    PSNR: 10.86 ± 2.65
```

## 推理

### 单张图像推理

```python
import torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline

# 加载模型
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "checkpoint_path",
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.to("cuda")

# 加载输入图像
input_image = Image.open("input.png")

# 生成预测
prompt = "pushing coffeemug from right to left"
output_image = pipe(
    prompt,
    image=input_image,
    num_inference_steps=50,
    image_guidance_scale=1.5
).images[0]

output_image.save("predicted_frame.png")
```

## 项目结构

```
dl5011/
├── README.md                          # 本文件
├── prepare_dataset.py                 # 数据处理脚本
├── frame_prediction_dataset.py        # 自定义Dataset类
├── train.py                           # 训练脚本
├── evaluate.py                        # 评估脚本
├── configs/
│   └── train_frame_prediction.yaml    # 训练配置
├── checkpoint/                        # 预训练模型
│   └── instruct-pix2pix-00-22000.ckpt
├── dataset/                           # 原始数据
│   ├── 20bn-something-something-v2/
│   └── labels/
├── processed_data/                    # 处理后的数据
├── logs/                              # 训练日志
└── evaluation_results/                # 评估结果
```

## 性能优化建议

### 内存优化

1. **降低批量大小**: 如果GPU内存不足，减小 `batch_size`
2. **使用半精度训练**: 配置文件中已启用 `precision: 16`
3. **梯度累积**: 已配置 `accumulate_grad_batches: 2`
4. **降低图像分辨率**: 使用 96×96 而不是 256×256

### 训练速度优化

1. **增加num_workers**: 根据CPU核心数调整
2. **使用多GPU**: 设置 `--gpus 0,1,2,3`
3. **减少验证频率**: 调整 `check_val_every_n_epoch`

## 常见问题

### 1. GPU内存不足

```python
# 减小批量大小
batch_size: 1

# 或使用CPU训练（非常慢）
--gpus ""  # 空字符串表示使用CPU
```

### 2. 数据集路径问题

确保配置文件中的路径正确:
```yaml
model:
  params:
    ckpt_path: /home/YueChang/phd_ws/dl5011/checkpoint/...

data:
  params:
    train:
      params:
        data_root: /home/YueChang/phd_ws/dl5011/processed_data
```

### 3. 模块导入错误

确保正确设置Python路径:
```python
sys.path.insert(0, '/home/YueChang/phd_ws/dl5011/instruct-pix2pix')
sys.path.insert(0, '/home/YueChang/phd_ws/dl5011/instruct-pix2pix/stable_diffusion')
```
