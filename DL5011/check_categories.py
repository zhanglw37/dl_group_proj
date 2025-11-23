"""
检查数据集中各任务类别的样本数量
"""

import json
from collections import defaultdict

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

def load_annotations(anno_path):
    """加载标注"""
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    return annotations

# 加载训练和验证集
train_annos = load_annotations('dataset/labels/train.json')
val_annos = load_annotations('dataset/labels/validation.json')

print(f"训练集: {len(train_annos)} 个样本")
print(f"验证集: {len(val_annos)} 个样本")
print(f"总计: {len(train_annos) + len(val_annos)} 个样本\n")

# 统计每个任务的样本数
all_annos = train_annos + val_annos

task_counts = defaultdict(int)
task_examples = defaultdict(list)

for anno in all_annos:
    template = anno.get('template', '')

    for task_name, templates in TASK_CATEGORIES.items():
        for target_template in templates:
            if template == target_template:
                task_counts[task_name] += 1
                if len(task_examples[task_name]) < 3:  # 只保存3个例子
                    task_examples[task_name].append(anno)
                break

print("=== 各任务类别样本统计 ===\n")
for task_name in ['move_object', 'drop_object', 'cover_object']:
    print(f"{task_name}: {task_counts[task_name]} 个样本")
    print("例子:")
    for ex in task_examples[task_name]:
        print(f"  - {ex['label']} (ID: {ex['id']})")
    print()

# 检查所有唯一的template
print("\n=== 检查可能遗漏的相关模板 ===")
all_templates = set()
for anno in all_annos:
    all_templates.add(anno.get('template', ''))

# 查找可能相关的模板
keywords = {
    'move_object': ['moving', 'push'],
    'drop_object': ['drop', 'fall', 'letting'],
    'cover_object': ['cover', 'putting', 'top']
}

for task_name, kws in keywords.items():
    print(f"\n{task_name} 相关的可能模板:")
    related = []
    for template in sorted(all_templates):
        if any(kw.lower() in template.lower() for kw in kws):
            # 检查是否已经在我们的列表中
            if template not in TASK_CATEGORIES[task_name]:
                related.append(template)

    for t in related[:10]:  # 只显示前10个
        print(f"  - {t}")
