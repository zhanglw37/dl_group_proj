"""
自定义Dataset类用于帧预测任务
用于InstructPix2Pix微调
第0帧 + 文本描述 → 预测第21帧
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FramePredictionDataset(Dataset):
    """
    Something-Something V2 帧预测数据集
    输入: 第0帧 + 文本描述
    输出: 第21帧
    """

    def __init__(
        self,
        data_root,
        split='train',
        size=256,
        interpolation="bicubic",
        flip_p=0.0
    ):
        """
        Args:
            data_root: 处理后的数据根目录
            split: 'train' 或 'val'
            size: 图像大小
            interpolation: 插值方法
            flip_p: 水平翻转概率（数据增强）
        """
        self.data_root = Path(data_root)
        self.split = split
        self.size = size
        self.flip_p = flip_p

        # 加载数据集信息
        info_path = self.data_root / 'dataset_info.json'
        with open(info_path, 'r') as f:
            self.dataset_info = json.load(f)

        # 合并所有任务的样本
        self.samples = []
        for task_name, task_samples in self.dataset_info[split].items():
            for sample in task_samples:
                sample['task'] = task_name
                self.samples.append(sample)

        print(f"加载 {split} 数据集: {len(self.samples)} 个样本")

        # 图像转换
        # PIL 9.0+ 使用Resampling枚举
        try:
            from PIL.Image import Resampling
            self.interpolation = {
                "linear": Resampling.BILINEAR,
                "bilinear": Resampling.BILINEAR,
                "bicubic": Resampling.BICUBIC,
                "lanczos": Resampling.LANCZOS,
            }[interpolation]
        except ImportError:
            # PIL < 9.0
            self.interpolation = {
                "linear": Image.BILINEAR,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC,
                "lanczos": Image.LANCZOS,
            }[interpolation]

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=self.interpolation),
            transforms.CenterCrop(size),
        ])

        self.tensor_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        input_path = self.data_root / sample['input_frame']
        target_path = self.data_root / sample['target_frame']

        input_image = Image.open(input_path).convert('RGB')
        target_image = Image.open(target_path).convert('RGB')

        # 数据增强：随机水平翻转
        if self.flip_p > 0.0 and torch.rand(1).item() < self.flip_p:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 应用变换
        input_image = self.image_transforms(input_image)
        target_image = self.image_transforms(target_image)

        # 转换为tensor
        input_tensor = self.tensor_transforms(input_image)
        target_tensor = self.tensor_transforms(target_image)

        # 文本提示 - 使用原始标签作为指令
        # 格式: "label" -> 例如 "pushing coffeemug from right to left"
        text_prompt = sample['label']

        # 返回格式需要匹配InstructPix2Pix的EditDataset
        # edited: 目标图像（第21帧）
        # edit: 包含c_concat（输入图像）和c_crossattn（文本提示）的字典
        return {
            'edited': target_tensor,     # [C, H, W] - 目标图像（第21帧）
            'edit': {
                'c_concat': input_tensor,  # [C, H, W] - 输入图像（第0帧）
                'c_crossattn': text_prompt  # 文本指令
            }
        }


if __name__ == "__main__":
    # 测试数据集
    dataset = FramePredictionDataset(
        data_root='processed_data',
        split='train',
        size=256
    )

    print(f"数据集大小: {len(dataset)}")
    print("\n第一个样本:")
    sample = dataset[0]
    print(f"  Input shape: {sample['jpg'].shape}")
    print(f"  Edited shape: {sample['edited'].shape}")
    print(f"  Text prompt: {sample['txt']}")
    print(f"  Task: {sample['task']}")
    print(f"  ID: {sample['id']}")

    # 检查值范围
    print(f"\n  Input range: [{sample['jpg'].min():.2f}, {sample['jpg'].max():.2f}]")
    print(f"  Edited range: [{sample['edited'].min():.2f}, {sample['edited'].max():.2f}]")
