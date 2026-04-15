#!/usr/bin/env python3
"""详细测试数据维度"""

import sys
sys.path.insert(0, '/home/wangpeng/CODE/RaCFormer-main')

import mmcv
mmcv.use_backend('cv2')

from mmcv import Config
from mmdet3d.datasets import build_dataset
import importlib
importlib.import_module('loaders')

print("\n" + "="*60)
print("详细测试数据维度")
print("="*60)

cfg = Config.fromfile('configs/own_racformer_r50_nuimg_704x256_f8.py')
dataset = build_dataset(cfg.data.test)

print(f"\n✓ 数据集大小: {len(dataset)}")

# 测试第15帧
data = dataset[15]
print("\n✓ 数据加载成功！")

# 详细检查
print("\n数据维度详情:")
print(f"  img: {data['img'][0].shape if isinstance(data['img'], list) else data['img'].data.shape}")
print(f"  radar_points: {len(data['radar_points'])} 个元素")
if len(data['radar_points']) > 0:
    print(f"    第一个元素: {data['radar_points'][0].shape if hasattr(data['radar_points'][0], 'shape') else type(data['radar_points'][0])}")
print(f"  radar_depth: {data['radar_depth'].shape}")
print(f"  radar_rcs: {data['radar_rcs'].shape}")
print(f"  gt_depth: {data['gt_depth'].shape}")

print("\n说明:")
print("  - img: 应该是 (48, 3, 256, 704) = 8时间帧 × 6相机")
print("  - radar_depth/rcs: 当前是 (6, 256, 704) = 6相机（仅当前帧）")
print("  - 简化配置：只使用当前帧radar，不使用历史帧")

print("\n" + "="*60)
print("✓ 数据加载测试完成！")
print("="*60)
