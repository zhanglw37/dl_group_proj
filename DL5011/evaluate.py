"""
评估脚本 - 计算SSIM和PSNR指标
"""

import sys
import os

# 添加instruct-pix2pix目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'instruct-pix2pix'))
sys.path.insert(0, os.path.join(script_dir, 'instruct-pix2pix', 'stable_diffusion'))
sys.path.insert(0, script_dir)

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from frame_prediction_dataset import FramePredictionDataset
from diffusers import StableDiffusionInstructPix2PixPipeline


def load_model(checkpoint_path, device='cuda'):
    """加载训练好的模型"""
    print(f"加载模型: {checkpoint_path}")

    # 使用diffusers库加载
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        safety_checker=None
    )
    pipe.to(device)

    return pipe


def tensor_to_numpy(tensor):
    """将tensor转换为numpy数组 (H, W, C)，值范围[0, 1]"""
    # tensor: (C, H, W), 值范围 [-1, 1]
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)
    img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    img = np.clip(img, 0, 1)
    return img


def calculate_metrics(pred_img, gt_img):
    """
    计算SSIM和PSNR
    Args:
        pred_img: numpy数组 (H, W, C), 值范围[0, 1]
        gt_img: numpy数组 (H, W, C), 值范围[0, 1]
    """
    # SSIM - multichannel for RGB
    ssim_value = ssim(gt_img, pred_img, multichannel=True, channel_axis=2, data_range=1.0)

    # PSNR
    psnr_value = psnr(gt_img, pred_img, data_range=1.0)

    return ssim_value, psnr_value


def evaluate(
    checkpoint_path,
    data_root,
    output_dir,
    split='val',
    num_samples=None,
    num_inference_steps=50,
    image_guidance_scale=1.5,
    device='cuda'
):
    """
    评估模型
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    dataset = FramePredictionDataset(
        data_root=data_root,
        split=split,
        size=256,
        flip_p=0.0
    )

    if num_samples is not None:
        dataset.samples = dataset.samples[:num_samples]

    print(f"评估样本数: {len(dataset)}")

    # 加载模型
    pipe = load_model(checkpoint_path, device=device)

    # 评估指标
    results = {
        'overall': {'ssim': [], 'psnr': []},
        'per_task': {}
    }

    # 逐样本评估
    for idx in tqdm(range(len(dataset)), desc="评估中"):
        sample = dataset[idx]

        input_tensor = sample['jpg']  # (C, H, W), [-1, 1]
        gt_tensor = sample['edited']  # (C, H, W), [-1, 1]
        text_prompt = sample['txt']
        task = sample['task']
        sample_id = sample['id']

        # 转换输入图像为PIL
        input_img_np = tensor_to_numpy(input_tensor)
        input_pil = Image.fromarray((input_img_np * 255).astype(np.uint8))

        # 生成预测
        with torch.no_grad():
            pred_pil = pipe(
                text_prompt,
                image=input_pil,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
            ).images[0]

        # 转换为numpy进行评估
        pred_np = np.array(pred_pil).astype(np.float32) / 255.0
        gt_np = tensor_to_numpy(gt_tensor)

        # 计算指标
        ssim_val, psnr_val = calculate_metrics(pred_np, gt_np)

        # 保存结果
        results['overall']['ssim'].append(ssim_val)
        results['overall']['psnr'].append(psnr_val)

        if task not in results['per_task']:
            results['per_task'][task] = {'ssim': [], 'psnr': []}
        results['per_task'][task]['ssim'].append(ssim_val)
        results['per_task'][task]['psnr'].append(psnr_val)

        # 保存可视化（可选，保存部分样本）
        if idx < 20:  # 只保存前20个样本
            save_dir = output_dir / task
            save_dir.mkdir(exist_ok=True)

            # 拼接图像：输入 | 预测 | 真值
            concat_img = Image.new('RGB', (256 * 3, 256))
            concat_img.paste(input_pil, (0, 0))
            concat_img.paste(pred_pil, (256, 0))
            concat_img.paste(Image.fromarray((gt_np * 255).astype(np.uint8)), (512, 0))
            concat_img.save(save_dir / f"{sample_id}_ssim{ssim_val:.3f}_psnr{psnr_val:.2f}.png")

    # 计算统计
    stats = {
        'overall': {
            'ssim_mean': float(np.mean(results['overall']['ssim'])),
            'ssim_std': float(np.std(results['overall']['ssim'])),
            'psnr_mean': float(np.mean(results['overall']['psnr'])),
            'psnr_std': float(np.std(results['overall']['psnr'])),
        },
        'per_task': {}
    }

    for task, metrics in results['per_task'].items():
        stats['per_task'][task] = {
            'ssim_mean': float(np.mean(metrics['ssim'])),
            'ssim_std': float(np.std(metrics['ssim'])),
            'psnr_mean': float(np.mean(metrics['psnr'])),
            'psnr_std': float(np.std(metrics['psnr'])),
            'num_samples': len(metrics['ssim'])
        }

    # 保存结果
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    print(f"\n总体:")
    print(f"  SSIM: {stats['overall']['ssim_mean']:.4f} ± {stats['overall']['ssim_std']:.4f}")
    print(f"  PSNR: {stats['overall']['psnr_mean']:.2f} ± {stats['overall']['psnr_std']:.2f}")

    print(f"\n各任务:")
    for task, task_stats in stats['per_task'].items():
        print(f"\n  {task} (n={task_stats['num_samples']}):")
        print(f"    SSIM: {task_stats['ssim_mean']:.4f} ± {task_stats['ssim_std']:.4f}")
        print(f"    PSNR: {task_stats['psnr_mean']:.2f} ± {task_stats['psnr_std']:.2f}")

    print(f"\n结果已保存至: {results_file}")
    print(f"可视化保存至: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Frame Prediction Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str,
                        default='processed_data',
                        help='Path to processed data')
    parser.add_argument('--output_dir', type=str,
                        default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                        help='Image guidance scale')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        num_inference_steps=args.steps,
        image_guidance_scale=args.guidance_scale,
        device=args.device
    )
