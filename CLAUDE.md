# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DL5011 course project that fine-tunes the InstructPix2Pix model for video frame prediction on the Something-Something V2 dataset. The task is to predict future frames (frame 21) from initial frames (frame 0 or 20) plus text descriptions of human-object interactions.

**Three task categories:**
- `move_object`: Moving/pushing objects left/right
- `drop_object`: Dropping/lifting objects
- `cover_object`: Covering/putting objects on top of others

**Dataset:** 284 training samples, 72 validation samples (24 per task), 256x256 resolution.

## Project Structure

```
DL5011/
├── instruct-pix2pix/           # Git submodule - original InstructPix2Pix implementation
├── configs/
│   ├── train_frame_prediction.yaml         # Main config: frame 0 -> 21
│   └── train_frame_prediction_20_21.yaml   # Alternative: frame 20 -> 21
├── dataset/                     # Raw video data & annotations
│   ├── 20bn-something-something-v2/
│   └── labels/
├── processed_data/              # Extracted frames (0->21 strategy)
├── processed_data_20_21/        # Extracted frames (20->21 strategy)
├── checkpoint/                  # Pre-trained InstructPix2Pix model
├── logs/                        # Training logs & checkpoints
└── Core scripts (see below)
```

## Key Python Modules

### Data Pipeline
- **`prepare_dataset.py`**: Extracts frames from videos, creates train/val splits
  - Defines `TASK_CATEGORIES` mapping task names to template strings
  - Supports two strategies: `frame0_21` (long-term) and `frame20_21` (short-term)
  - Creates `dataset_info.json` with sample metadata

- **`frame_prediction_dataset.py`**: PyTorch Dataset for training
  - `FramePredictionDataset`: Returns dict with `edited` (target frame) and `edit` dict containing `c_concat` (input image) and `c_crossattn` (text prompt)
  - Loads from processed data using `dataset_info.json`
  - Normalizes images to [-1, 1] range

### Training & Evaluation
- **`train.py`**: Training script using PyTorch Lightning
  - Uses `omegaconf` to load config YAML
  - Instantiates model and data via `instantiate_from_config()`
  - Creates Lightning Trainer with ModelCheckpoint, LearningRateMonitor callbacks
  - Hardcoded sys.path insertions for `instruct-pix2pix` submodule

- **`evaluate_lightning.py`**: Evaluation from Lightning checkpoints
  - Loads model using config + checkpoint
  - Uses DDIM sampling for generation
  - Calculates SSIM and PSNR metrics per task
  - Saves visualizations as input|prediction|ground_truth triplets

## Common Commands

### Environment Setup
```bash
cd DL5011/instruct-pix2pix
conda env create -f environment.yaml
conda activate ip2p
pip install scikit-image diffusers
```

### Data Preparation
```bash
# Strategy 1: Frame 0 -> 21 (long-term prediction)
python prepare_dataset.py \
    --max_samples 120 \
    --output_dir processed_data \
    --size 256 \
    --strategy frame0_21

# Strategy 2: Frame 20 -> 21 (short-term prediction)
python prepare_dataset.py \
    --max_samples 120 \
    --output_dir processed_data_20_21 \
    --size 256 \
    --strategy frame20_21
```

### Training
```bash
# Train with frame 0 -> 21 strategy
python train.py \
    --config configs/train_frame_prediction.yaml \
    --gpus 0 \
    --seed 42

# Train with frame 20 -> 21 strategy
python train.py \
    --config configs/train_frame_prediction_20_21.yaml \
    --gpus 0 \
    --seed 42

# Resume from checkpoint
python train.py \
    --config configs/train_frame_prediction.yaml \
    --resume logs/frame_prediction_XXXXXXXX_XXXXXX/checkpoints/last.ckpt \
    --gpus 0
```

### Monitoring Training
```bash
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

### Evaluation
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

## Architecture Notes

### Model Architecture
- Based on Stable Diffusion with InstructPix2Pix modifications
- **Conditioning**: Hybrid conditioning with concatenated input image + cross-attention text
- **Input**: 8 channels (4 latent channels from input image + 4 channels from noise)
- **Output**: 4 latent channels decoded to RGB image
- **Text encoder**: Frozen CLIP embeddings
- **VAE**: Encodes/decodes between pixel space (256x256) and latent space (32x32)
- **UNet**: 320 base channels, attention at resolutions [4,2,1], transformer depth 1

### Training Configuration
- Learning rate: 5e-5 (lower for fine-tuning)
- Batch size: 2, gradient accumulation: 2 (effective batch size: 4)
- Precision: 32-bit (can use 16 for memory savings)
- EMA disabled to simplify training
- Validation every 2 epochs
- Checkpoint saves top 3 models by `val/loss_simple_ema`

### Critical Path Dependencies
The code has hardcoded paths that need adjustment:
1. `train.py` lines 10-12: `sys.path.insert()` for instruct-pix2pix modules
2. `evaluate_lightning.py` lines 7-9: Same path insertions
3. Config files: `ckpt_path`, `data_root` use absolute paths starting with `/home/YueChang/phd_ws/dl5011/`

When working on a different machine, update these paths to match your environment.

### Data Format
- Input images: PNG files `{video_id}_input.png`
- Target images: PNG files `{video_id}_target.png`
- Metadata: `dataset_info.json` contains split -> task -> samples mapping
- Each sample has: `id`, `label`, `template`, `input_frame`, `target_frame`

## Git Submodules

The `instruct-pix2pix` directory is a git submodule. To initialize:
```bash
git submodule update --init --recursive
```

## Important Constraints

1. **Horizontal flips disabled**: `flip_p=0.0` in configs because flipping changes action semantics (e.g., "left to right" becomes incorrect)
2. **Fixed image size**: 256x256 in production configs (96x96 was tested for faster iteration)
3. **No classifier-free guidance during training**: Model uses hybrid conditioning only
4. **DDIM sampling**: 50 steps used for evaluation (configurable via `--steps`)
