"""
数据集准备脚本 - Something-Something V2 for Frame Prediction
从视频中提取帧，并按照任务类别筛选数据
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


# 定义三个任务类别对应的模板（使用占位符格式）
TASK_CATEGORIES = {
    'move_object': [
        'Moving [something] from left to right',
        'Moving [something] from right to left',
        'Pushing [something] from left to right',
        'Pushing [something] from right to left',
    ],
    'drop_object': [
        'Dropping [something] onto [something]',
        'Lifting up one end of [something], then letting it drop down',
        'Lifting [something] up completely, then letting it drop down',
    ],
    'cover_object': [
        'Covering [something] with [something]',
        'Putting [something] on top of [something]',
        'Putting [something] onto [something]',
    ]
}


def load_labels(labels_path):
    """加载标签映射"""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def load_annotations(anno_path):
    """加载训练/验证标注"""
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def filter_by_task(annotations, task_categories):
    """根据任务类别筛选数据"""
    filtered_data = {task: [] for task in task_categories.keys()}

    for anno in annotations:
        template = anno.get('template', '')

        for task_name, templates in task_categories.items():
            # 检查template是否匹配任一目标模板
            for target_template in templates:
                if template == target_template:
                    filtered_data[task_name].append(anno)
                    break

    return filtered_data


def extract_frames_from_video(video_path, frame_indices=[0, 21]):
    """
    从视频中提取指定帧
    Args:
        video_path: 视频文件路径
        frame_indices: 要提取的帧索引列表，默认[0, 21]表示第0帧和第21帧
    Returns:
        frames: 提取的帧字典 {frame_idx: frame_array}
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 确保请求的帧索引在有效范围内
    valid_indices = [idx for idx in frame_indices if idx < total_frames]

    if len(valid_indices) != len(frame_indices):
        print(f"警告: 视频 {video_path} 只有 {total_frames} 帧，请求的帧索引: {frame_indices}")

    frames = {}
    for idx in valid_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames[idx] = frame
        else:
            print(f"无法读取视频 {video_path} 的第 {idx} 帧")

    cap.release()
    return frames


def prepare_dataset(
    video_dir,
    labels_path,
    train_anno_path,
    val_anno_path,
    output_dir,
    max_samples_per_task=100,
    target_size=(96, 96),
    strategy='frame0_21'
):
    """
    准备训练数据集
    Args:
        video_dir: 视频文件目录
        labels_path: 标签映射文件
        train_anno_path: 训练集标注文件
        val_anno_path: 验证集标注文件
        output_dir: 输出目录
        max_samples_per_task: 每个任务最多采样数量
        target_size: 目标图像大小 (height, width)
        strategy: 预测策略 'frame0_21' 或 'frame20_21'
    """

    # 根据策略确定要提取的帧
    if strategy == 'frame0_21':
        input_frame_idx = 0
        target_frame_idx = 21
        print(f"使用策略: 第{input_frame_idx}帧 -> 第{target_frame_idx}帧")
    elif strategy == 'frame20_21':
        input_frame_idx = 20
        target_frame_idx = 21
        print(f"使用策略: 第{input_frame_idx}帧 -> 第{target_frame_idx}帧")
    else:
        raise ValueError(f"未知策略: {strategy}")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 为每个任务创建子目录
    for task_name in TASK_CATEGORIES.keys():
        (output_dir / task_name).mkdir(exist_ok=True)

    # 加载标注
    print("加载标注文件...")
    train_annos = load_annotations(train_anno_path)
    val_annos = load_annotations(val_anno_path)
    all_annos = train_annos + val_annos

    print(f"总共 {len(all_annos)} 个样本")

    # 按任务筛选
    print("按任务类别筛选数据...")
    filtered_data = filter_by_task(all_annos, TASK_CATEGORIES)

    # 打印每个任务的样本数
    for task_name, samples in filtered_data.items():
        print(f"{task_name}: {len(samples)} 个样本")

    # 准备数据集元信息
    dataset_info = {
        'train': {},
        'val': {}
    }

    # 处理每个任务
    for task_name, samples in filtered_data.items():
        print(f"\n处理任务: {task_name}")

        # 限制样本数量
        if len(samples) > max_samples_per_task:
            # 随机采样
            np.random.seed(42)
            samples = np.random.choice(samples, max_samples_per_task, replace=False).tolist()
            print(f"随机采样 {max_samples_per_task} 个样本")

        task_samples = []
        success_count = 0

        for idx, sample in enumerate(tqdm(samples, desc=f"提取 {task_name} 帧")):
            video_id = sample['id']
            video_filename = f"{video_id}.webm"
            video_path = Path(video_dir) / video_filename

            if not video_path.exists():
                print(f"视频不存在: {video_path}")
                continue

            # 提取输入帧和目标帧
            frames = extract_frames_from_video(video_path, frame_indices=[input_frame_idx, target_frame_idx])

            if frames is None or input_frame_idx not in frames or target_frame_idx not in frames:
                continue

            # 调整大小
            input_frame = cv2.resize(frames[input_frame_idx], target_size)
            target_frame = cv2.resize(frames[target_frame_idx], target_size)

            # 保存帧
            input_path = output_dir / task_name / f"{video_id}_input.png"
            target_path = output_dir / task_name / f"{video_id}_target.png"

            cv2.imwrite(str(input_path), input_frame)
            cv2.imwrite(str(target_path), target_frame)

            # 记录样本信息
            task_samples.append({
                'id': video_id,
                'label': sample['label'],
                'template': sample['template'],
                'input_frame': str(input_path.relative_to(output_dir)),
                'target_frame': str(target_path.relative_to(output_dir)),
            })

            success_count += 1

        print(f"{task_name}: 成功处理 {success_count}/{len(samples)} 个样本")

        # 划分训练集和验证集 (80/20)
        split_idx = int(0.8 * len(task_samples))
        dataset_info['train'][task_name] = task_samples[:split_idx]
        dataset_info['val'][task_name] = task_samples[split_idx:]

    # 保存数据集信息
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f"\n数据集准备完成！信息保存至: {info_path}")

    # 打印统计信息
    print("\n=== 数据集统计 ===")
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        total = 0
        for task_name, samples in dataset_info[split].items():
            print(f"  {task_name}: {len(samples)} 个样本")
            total += len(samples)
        print(f"  总计: {total} 个样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='准备 Something-Something V2 数据集用于帧预测')
    parser.add_argument('--video_dir', type=str,
                        default='dataset/20bn-something-something-v2',
                        help='视频文件目录')
    parser.add_argument('--labels_path', type=str,
                        default='dataset/labels/labels.json',
                        help='标签映射文件')
    parser.add_argument('--train_anno', type=str,
                        default='dataset/labels/train.json',
                        help='训练集标注文件')
    parser.add_argument('--val_anno', type=str,
                        default='dataset/labels/validation.json',
                        help='验证集标注文件')
    parser.add_argument('--output_dir', type=str,
                        default='processed_data',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=150,
                        help='每个任务最多采样数量')
    parser.add_argument('--size', type=int, default=96,
                        help='图像尺寸 (正方形)')
    parser.add_argument('--strategy', type=str, default='frame0_21',
                        choices=['frame0_21', 'frame20_21'],
                        help='预测策略: frame0_21(第0帧预测第21帧) 或 frame20_21(第20帧预测第21帧)')

    args = parser.parse_args()

    prepare_dataset(
        video_dir=args.video_dir,
        labels_path=args.labels_path,
        train_anno_path=args.train_anno,
        val_anno_path=args.val_anno,
        output_dir=args.output_dir,
        max_samples_per_task=args.max_samples,
        target_size=(args.size, args.size),
        strategy=args.strategy
    )
