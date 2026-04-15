#!/usr/bin/env python3
"""简单测试数据加载"""

import sys
sys.path.insert(0, '/home/wangpeng/CODE/RaCFormer-main')

import mmcv
mmcv.use_backend('cv2')

from mmcv import Config
from mmdet3d.datasets import build_dataset
import importlib
importlib.import_module('loaders')

print("\n" + "="*60)
print("测试自定义数据集加载")
print("="*60)

# 加载配置
cfg = Config.fromfile('configs/own_racformer_r50_nuimg_704x256_f8.py')
dataset = build_dataset(cfg.data.test)

print(f"\n✓ 数据集大小: {len(dataset)}")

# 测试加载第15帧
print("\n测试加载第15帧...")
data = dataset[15]

print("✓ 数据加载成功！")
print(f"\n数据keys: {list(data.keys())}")

# 检查维度
for key in ['img', 'radar_points', 'radar_depth', 'radar_rcs', 'gt_depth']:
    if key in data:
        val = data[key]
        if hasattr(val, 'data'):
            val = val.data
        if isinstance(val, list) and len(val) > 0:
            val = val[0]
        if hasattr(val, 'shape'):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")

print("\n" + "="*60)
print("✓ 测试通过！")
print("="*60)
