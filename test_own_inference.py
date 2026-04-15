#!/usr/bin/env python3
"""
自定义数据推理脚本（无需标注，仅使用GPU 0）

用法:
    python test_own_inference.py \
        --config configs/own_racformer_r50_nuimg_704x256_f8.py \
        --weights checkpoints/racformer_r50_f8.pth \
        --out results/own_data_results.pkl \
        --vis_dir results/visualizations
"""

import os
import sys
import pickle
import argparse
import importlib
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='自定义数据推理脚本')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--weights', required=True, help='模型权重路径')
    parser.add_argument('--out', default='results/own_data_results.pkl', help='输出结果文件')
    parser.add_argument('--vis_dir', default=None, help='可视化输出目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID（默认0）')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # 强制使用指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.cuda.set_device(0)  # 设置为第一个可见GPU（即GPU 0）
    
    print("\n" + "="*60)
    print("自定义数据推理脚本")
    print("="*60)
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载配置
    print(f"\n加载配置文件: {args.config}")
    cfg = Config.fromfile(args.config)
    
    # 注册自定义模块
    print("注册自定义模块...")
    importlib.import_module('models')
    importlib.import_module('loaders')
    
    # 设置随机种子
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True
    
    # 构建数据集
    print(f"\n{'='*60}")
    print("构建数据集...")
    print(f"{'='*60}")
    dataset = build_dataset(cfg.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,  # 单GPU
        dist=False,
        shuffle=False,
        seed=0,
    )
    print(f"✓ 数据集大小: {len(dataset)}")
    print(f"✓ 批大小: {args.batch_size}")
    print(f"✓ Worker数量: {cfg.data.workers_per_gpu}")
    
    # 构建模型
    print(f"\n{'='*60}")
    print("构建模型...")
    print(f"{'='*60}")
    print(f"模型类型: {cfg.model.type}")
    model = build_model(cfg.model)
    model.cuda()
    model = MMDataParallel(model, [0])  # 只使用GPU 0
    print(f"✓ 模型已创建并移至GPU 0")
    
    # 加载权重
    print(f"\n{'='*60}")
    print("加载模型权重...")
    print(f"{'='*60}")
    print(f"权重文件: {args.weights}")
    checkpoint = load_checkpoint(
        model, 
        args.weights, 
        map_location='cuda',
        strict=False  # 允许部分权重不匹配
    )
    print(f"✓ 权重加载成功")
    
    if 'version' in checkpoint:
        print(f"✓ 模型版本: {checkpoint.get('version', 'unknown')}")
    
    # 推理
    print(f"\n{'='*60}")
    print("开始推理...")
    print(f"{'='*60}")
    model.eval()
    
    results = single_gpu_test(model, dataloader, show=False)
    
    # 保存结果
    print(f"\n{'='*60}")
    print("保存结果...")
    print(f"{'='*60}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ 结果已保存到: {args.out}")
    
    # 统计检测结果
    total_detections = 0
    class_counts = {}
    
    for result in results:
        if 'boxes_3d' in result:
            num_boxes = len(result['boxes_3d'])
            total_detections += num_boxes
            
            if 'labels_3d' in result:
                for label in result['labels_3d'].cpu().numpy():
                    class_counts[int(label)] = class_counts.get(int(label), 0) + 1
    
    print(f"✓ 共处理 {len(results)} 帧")
    print(f"✓ 检测到 {total_detections} 个目标")
    
    if class_counts:
        print(f"\n各类别统计:")
        class_names = cfg.class_names
        for class_id, count in sorted(class_counts.items()):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            print(f"  {class_name}: {count} 个")
    
    # 可视化（如果指定）
    if args.vis_dir:
        print(f"\n{'='*60}")
        print("生成可视化...")
        print(f"{'='*60}")
        print(f"可视化目录: {args.vis_dir}")
        print("提示: 可视化功能需要单独运行 tools/visualize_results.py")
        print(f"命令: python tools/visualize_results.py --results {args.out} --vis_dir {args.vis_dir}")
    
    print(f"\n{'='*60}")
    print("✓ 推理完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
