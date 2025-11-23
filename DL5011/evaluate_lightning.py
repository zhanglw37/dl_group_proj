"""
评估脚本 - 从PyTorch Lightning checkpoint加载模型
计算SSIM和PSNR指标
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
from omegaconf import OmegaConf

from frame_prediction_dataset import FramePredictionDataset
from ldm.util import instantiate_from_config


def load_model_from_checkpoint(config_path, checkpoint_path, device='cuda'):
    """从Lightning checkpoint加载模型"""
    print(f"加载配置: {config_path}")
    config = OmegaConf.load(config_path)

    print(f"加载checkpoint: {checkpoint_path}")
    model = instantiate_from_config(config.model)

    # 加载checkpoint权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    return model


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
    ssim_value = ssim(gt_img, pred_img, channel_axis=2, data_range=1.0)

    # PSNR
    psnr_value = psnr(gt_img, pred_img, data_range=1.0)

    return ssim_value, psnr_value


@torch.no_grad()
def evaluate(
    config_path,
    checkpoint_path,
    data_root,
    output_dir,
    split='val',
    num_samples=None,
    num_inference_steps=50,
    cfg_scale=7.5,
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
    model = load_model_from_checkpoint(config_path, checkpoint_path, device=device)

    # 评估指标
    results = {
        'overall': {'ssim': [], 'psnr': []},
        'per_task': {}
    }

    # 逐样本评估
    for idx in tqdm(range(len(dataset)), desc="评估中"):
        sample = dataset[idx]

        # 获取数据
        input_tensor = sample['edit']['c_concat'].unsqueeze(0).to(device)  # (1, C, H, W)
        gt_tensor = sample['edited']  # (C, H, W)
        text_prompt = sample['edit']['c_crossattn']

        # 这里需要从原始数据集获取task和id信息
        # 因为经过数据集返回后这些信息丢失了
        # 我们需要修改数据集以保留这些信息
        task = dataset.samples[idx]['task']
        sample_id = dataset.samples[idx]['id']

        # 使用模型的采样方法生成预测
        # 这里我们使用DDIM采样
        with model.ema_scope():
            # 编码文本条件
            c = model.get_learned_conditioning([text_prompt])

            # 编码图像条件到latent空间
            c_cat = model.encode_first_stage(input_tensor.to(device)).mode()

            # 准备条件字典 - 格式需要匹配训练时的格式
            cond = {
                "c_concat": [c_cat],
                "c_crossattn": [c],
            }

            # 采样
            shape = [4, 32, 32]  # latent shape
            samples, _ = model.sample_log(
                cond=cond,
                batch_size=1,
                ddim=True,
                ddim_steps=num_inference_steps,
                eta=0.0
            )

            # 解码
            x_samples = model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

        # 转换为numpy
        pred_np = x_samples[0].permute(1, 2, 0).cpu().numpy()
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

        # 保存可视化（前20个样本）
        if idx < 20:
            save_dir = output_dir / task
            save_dir.mkdir(exist_ok=True)

            # 转换为PIL图像
            input_np = tensor_to_numpy(input_tensor[0].cpu())
            input_pil = Image.fromarray((input_np * 255).astype(np.uint8))
            pred_pil = Image.fromarray((pred_np * 255).astype(np.uint8))
            gt_pil = Image.fromarray((gt_np * 255).astype(np.uint8))

            # 拼接图像：输入 | 预测 | 真值
            concat_img = Image.new('RGB', (256 * 3, 256))
            concat_img.paste(input_pil, (0, 0))
            concat_img.paste(pred_pil, (256, 0))
            concat_img.paste(gt_pil, (512, 0))
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
    parser = argparse.ArgumentParser(description='Evaluate Frame Prediction Model from Lightning Checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to training config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Lightning checkpoint')
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
                        help='Number of DDIM sampling steps')
    parser.add_argument('--cfg_scale', type=float, default=7.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        split=args.split,
        num_samples=args.num_samples,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg_scale,
        device=args.device
    )
