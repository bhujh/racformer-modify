#!/usr/bin/env python3
"""
测试自定义数据集加载

用法:
    python tools/test_dataset_loading.py
"""

import sys
sys.path.insert(0, '/home/wangpeng/CODE/RaCFormer-main')

import mmcv
# 设置使用cv2后端以支持PNG文件
mmcv.use_backend('cv2')

from mmcv import Config
from mmdet3d.datasets import build_dataset

# 重要：导入loaders模块以注册自定义数据集
import importlib
importlib.import_module('loaders')

def test_dataset_loading():
    """测试数据集加载"""
    print("\n" + "=" * 60)
    print("测试自定义数据集加载")
    print("=" * 60)
    
    # 加载配置
    print("\n1. 加载配置文件...")
    cfg = Config.fromfile('configs/own_racformer_r50_nuimg_704x256_f8.py')
    print("✓ 配置文件加载成功")
    
    # 构建数据集
    print("\n2. 构建数据集...")
    dataset = build_dataset(cfg.data.test)
    print(f"✓ 数据集构建成功")
    print(f"  数据集类型: {type(dataset).__name__}")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  建议测试范围: 第14-101帧 (共88帧)")
    
    # 测试加载第15帧（确保有足够历史帧）
    print("\n3. 测试加载第15帧...")
    try:
        data = dataset[15]
        print("✓ 数据加载成功")
        
        # 检查数据keys
        print(f"\n4. 数据keys: {list(data.keys())}")
        
        # 检查数据维度
        print("\n5. 数据维度检查:")
        if 'img' in data:
            print(f"  ✓ img: {data['img'].data.shape}")
            expected_img_shape = (48, 3, 256, 704)  # 8帧 × 6相机
            if data['img'].data.shape == expected_img_shape:
                print(f"    ✓ 维度正确！期望 {expected_img_shape}")
            else:
                print(f"    ✗ 维度不匹配！期望 {expected_img_shape}")
        
        if 'radar_points' in data:
            print(f"  ✓ radar_points: {type(data['radar_points'])}")
            if isinstance(data['radar_points'], list):
                print(f"    长度: {len(data['radar_points'])}")
                if len(data['radar_points']) > 0:
                    print(f"    第一个元素类型: {type(data['radar_points'][0])}")
        
        if 'radar_depth' in data:
            print(f"  ✓ radar_depth: {data['radar_depth'].data.shape}")
            expected_depth_shape = (48, 1, 256, 704)
            if data['radar_depth'].data.shape == expected_depth_shape:
                print(f"    ✓ 维度正确！期望 {expected_depth_shape}")
            else:
                print(f"    ✗ 维度不匹配！期望 {expected_depth_shape}")
        
        if 'radar_rcs' in data:
            print(f"  ✓ radar_rcs: {data['radar_rcs'].data.shape}")
            expected_rcs_shape = (48, 1, 256, 704)
            if data['radar_rcs'].data.shape == expected_rcs_shape:
                print(f"    ✓ 维度正确！期望 {expected_rcs_shape}")
            else:
                print(f"    ✗ 维度不匹配！期望 {expected_rcs_shape}")
        
        if 'gt_depth' in data:
            print(f"  ✓ gt_depth: {data['gt_depth'].data.shape}")
        
        # 检查元数据
        if 'img_metas' in data:
            print(f"\n6. 元数据检查:")
            img_metas = data['img_metas'].data
            print(f"  ✓ img_metas类型: {type(img_metas)}")
            if isinstance(img_metas, dict):
                print(f"  ✓ img_metas keys: {list(img_metas.keys())}")
                if 'filename' in img_metas:
                    print(f"  ✓ filename数量: {len(img_metas['filename'])}")
                if 'lidar2img' in img_metas:
                    print(f"  ✓ lidar2img数量: {len(img_metas['lidar2img'])}")
        
        print("\n" + "=" * 60)
        print("✓ 数据加载测试通过！")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ 数据加载失败！")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_dataset_loading()
    sys.exit(0 if success else 1)
