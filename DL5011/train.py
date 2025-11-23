"""
训练脚本 - 微调InstructPix2Pix用于帧预测
基于 instruct-pix2pix/main.py 简化
"""

import sys
import os

# 添加instruct-pix2pix目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'instruct-pix2pix'))
sys.path.insert(0, os.path.join(script_dir, 'instruct-pix2pix', 'stable_diffusion'))
sys.path.insert(0, script_dir)

import argparse
import datetime
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser(description='Train Frame Prediction Model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_frame_prediction.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='GPU IDs to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='logs',
        help='Directory for logs'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # 设置随机种子
    seed_everything(args.seed)

    # 加载配置
    config = OmegaConf.load(args.config)

    # 实例化数据模块
    data = instantiate_from_config(config.data)
    data.setup()

    print(f"训练集大小: {len(data.train_dataloader().dataset)}")
    print(f"验证集大小: {len(data.val_dataloader().dataset)}")

    # 实例化模型
    model = instantiate_from_config(config.model)

    # 配置学习率
    bs = config.data.params.batch_size
    base_lr = config.model.base_learning_rate
    ngpu = len([int(x) for x in args.gpus.split(',')]) if args.gpus else 1
    accumulate_grad_batches = config.lightning.trainer.get('accumulate_grad_batches', 1)

    # 不使用LR缩放，直接使用base_lr
    model.learning_rate = base_lr
    print(f"设置学习率: {model.learning_rate:.2e}")
    print(f"批量大小: {bs}")
    print(f"梯度累积: {accumulate_grad_batches}")
    print(f"有效批量大小: {bs * accumulate_grad_batches}")

    # 设置日志目录
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(args.logdir, f"frame_prediction_{now}")
    os.makedirs(logdir, exist_ok=True)

    # TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=logdir,
        name="tensorboard"
    )

    # 回调函数
    callbacks = []

    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logdir, "checkpoints"),
        filename="frame-pred-{epoch:02d}-{val/loss_simple_ema:.4f}",
        monitor="val/loss_simple_ema",
        save_top_k=3,
        mode="min",
        save_last=True,
        every_n_epochs=2
    )
    callbacks.append(checkpoint_callback)

    # LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # ImageLogger (如果配置中有)
    if "callbacks" in config.lightning and "image_logger" in config.lightning.callbacks:
        image_logger = instantiate_from_config(config.lightning.callbacks.image_logger)
        callbacks.append(image_logger)

    # 训练器配置
    trainer_config = config.lightning.trainer

    # GPU设置
    gpu_ids = [int(x) for x in args.gpus.split(',')]

    # 创建Trainer
    trainer = Trainer(
        devices=gpu_ids,
        accelerator="gpu" if len(gpu_ids) > 0 else "cpu",
        max_epochs=trainer_config.max_epochs,
        precision=trainer_config.get('precision', 32),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 1),
        logger=logger,
        callbacks=callbacks,
        benchmark=trainer_config.get('benchmark', True),
        log_every_n_steps=50,
    )

    # 开始训练
    print(f"\n{'='*50}")
    print(f"开始训练 - 日志目录: {logdir}")
    print(f"GPU: {gpu_ids}")
    print(f"批量大小: {config.data.params.batch_size}")
    print(f"累积梯度: {trainer_config.get('accumulate_grad_batches', 1)}")
    print(f"有效批量大小: {config.data.params.batch_size * trainer_config.get('accumulate_grad_batches', 1)}")
    print(f"学习率: {config.model.base_learning_rate}")
    print(f"最大轮数: {trainer_config.max_epochs}")
    print(f"{'='*50}\n")

    if args.resume:
        trainer.fit(model, data, ckpt_path=args.resume)
    else:
        trainer.fit(model, data)

    print(f"\n训练完成！检查点保存在: {os.path.join(logdir, 'checkpoints')}")


if __name__ == "__main__":
    main()
