#!/usr/bin/env python3
"""
生成自定义数据集的元数据文件 (own_infos.pkl)

用法:
    python tools/create_own_data.py
"""

import os
import numpy as np
import pickle
from glob import glob
from pyquaternion import Quaternion

# 配置
DATA_ROOT = "/mnt/diskNvme0/wangpengData/OwnData/"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
RADAR_DIR = os.path.join(DATA_ROOT, "radar")
CALIB_DIR = os.path.join(DATA_ROOT, "calibration")
OUTPUT_FILE = os.path.join(DATA_ROOT, "own_infos.pkl")

# 采集参数
FRAME_RATE = 15  # Hz
FRAME_INTERVAL_US = int(1e6 / FRAME_RATE)  # 微秒 (66667)


def load_calibration():
    """加载标定文件"""
    print("\n加载标定文件...")
    
    # 读取相机内参
    K = np.loadtxt(os.path.join(CALIB_DIR, "camera_intrinsic.txt"))
    print(f"✓ 相机内参 K:\n{K}")
    
    # 读取外参：Radar到Camera
    T_r2c = np.loadtxt(os.path.join(CALIB_DIR, "radar_to_camera.txt"))
    print(f"\n✓ Radar到Camera变换矩阵:\n{T_r2c}")
    
    # 转换为：Camera到Radar（用作sensor2lidar）
    T_c2r = np.linalg.inv(T_r2c)
    R_c2r = T_c2r[:3, :3]
    t_c2r = T_c2r[:3, 3]
    
    print(f"\n✓ Camera到Radar旋转矩阵:\n{R_c2r}")
    print(f"✓ Camera到Radar平移向量: {t_c2r}")
    
    return K, R_c2r, t_c2r, T_r2c


def create_camera_info(image_path, timestamp, K, R_c2r, t_c2r):
    """创建单个相机的信息字典"""
    return {
        'data_path': image_path,
        'timestamp': timestamp,
        'sensor2lidar_rotation': R_c2r.copy(),
        'sensor2lidar_translation': t_c2r.copy(),
        'cam_intrinsic': K.copy(),
        'sensor2global_rotation': R_c2r.copy(),  # 简化：假设radar为全局坐标系
        'sensor2global_translation': t_c2r.copy(),
    }


def create_data_infos():
    """创建完整的数据信息列表"""
    print("\n" + "=" * 60)
    print("开始生成数据信息...")
    print("=" * 60)
    
    # 加载标定
    K, R_c2r, t_c2r, T_r2c = load_calibration()
    
    # 扫描文件
    print(f"\n扫描数据文件...")
    image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))  # 改为jpg
    if len(image_files) == 0:
        # 如果没有jpg，尝试png
        image_files = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    radar_files = sorted(glob(os.path.join(RADAR_DIR, "*.pcd")))
    
    print(f"✓ 找到 {len(image_files)} 个图像文件")
    print(f"✓ 找到 {len(radar_files)} 个radar文件")
    
    # 确保数量一致
    if len(image_files) != len(radar_files):
        raise ValueError(f"图像和radar数量不匹配: {len(image_files)} vs {len(radar_files)}")
    
    # 生成数据信息
    print(f"\n生成元数据...")
    data_infos = []
    
    for idx, (img_file, radar_file) in enumerate(zip(image_files, radar_files)):
        # 生成token（7位数字字符串）
        token = f"{idx:07d}"
        
        # 生成时间戳（微秒）
        timestamp = idx * FRAME_INTERVAL_US
        
        # 相对路径
        img_rel_path = os.path.relpath(img_file, DATA_ROOT)
        radar_rel_path = os.path.relpath(radar_file, DATA_ROOT)
        
        # 创建相机信息（复制6份，模拟多相机配置）
        cam_info = create_camera_info(img_rel_path, timestamp, K, R_c2r, t_c2r)
        cams = {
            'CAM_FRONT': cam_info.copy(),
            'CAM_FRONT_RIGHT': cam_info.copy(),
            'CAM_FRONT_LEFT': cam_info.copy(),
            'CAM_BACK': cam_info.copy(),
            'CAM_BACK_LEFT': cam_info.copy(),
            'CAM_BACK_RIGHT': cam_info.copy(),
        }
        
        # 创建radar信息
        rads = {
            'RADAR_FRONT': {
                'data_path': radar_rel_path,
                'timestamp': timestamp,
            }
        }
        
        # 创建单帧信息
        info = {
            'token': token,
            'timestamp': timestamp,
            'lidar_path': radar_rel_path,  # 使用radar路径代替lidar
            
            # 坐标系变换（简化：假设radar为静态参考系）
            'ego2global_translation': np.array([0.0, 0.0, 0.0]),
            'ego2global_rotation': np.array([1.0, 0.0, 0.0, 0.0]),  # 四元数 [w,x,y,z]
            'lidar2ego_translation': np.array([0.0, 0.0, 0.0]),
            'lidar2ego_rotation': np.array([1.0, 0.0, 0.0, 0.0]),
            
            'cams': cams,
            'rads': rads,
            'sweeps': [],  # 由collect_sweeps动态生成
        }
        
        data_infos.append(info)
        
        if (idx + 1) % 20 == 0:
            print(f"  已处理 {idx + 1}/{len(image_files)} 帧")
    
    print(f"\n✓ 总共生成 {len(data_infos)} 帧数据信息")
    return data_infos


def save_data_infos(data_infos, output_file):
    """保存数据信息到pkl文件"""
    print(f"\n保存到 {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(data_infos, f)
    
    # 检查文件大小
    file_size = os.path.getsize(output_file) / 1024 / 1024  # MB
    print(f"✓ 文件大小: {file_size:.2f} MB")
    print("✓ 保存完成！")


def verify_data_infos(pkl_file):
    """验证生成的pkl文件"""
    print("\n" + "=" * 60)
    print(f"验证 {pkl_file}...")
    print("=" * 60)
    
    with open(pkl_file, 'rb') as f:
        data_infos = pickle.load(f)
    
    print(f"\n✓ 总帧数: {len(data_infos)}")
    print(f"✓ 第一帧keys: {list(data_infos[0].keys())}")
    print(f"✓ 相机数量: {len(data_infos[0]['cams'])}")
    print(f"✓ Radar数量: {len(data_infos[0]['rads'])}")
    print(f"✓ 第一帧时间戳: {data_infos[0]['timestamp']} 微秒")
    print(f"✓ 最后一帧时间戳: {data_infos[-1]['timestamp']} 微秒")
    
    # 检查文件路径
    print(f"\n示例文件路径:")
    print(f"  图像: {data_infos[0]['cams']['CAM_FRONT']['data_path']}")
    print(f"  Radar: {data_infos[0]['lidar_path']}")
    
    # 检查坐标系变换
    print(f"\n坐标系变换信息:")
    print(f"  sensor2lidar_rotation shape: {data_infos[0]['cams']['CAM_FRONT']['sensor2lidar_rotation'].shape}")
    print(f"  sensor2lidar_translation shape: {data_infos[0]['cams']['CAM_FRONT']['sensor2lidar_translation'].shape}")
    
    # 计算可用测试范围
    sweep_interval = 2
    sweeps_num = 7
    min_frame = sweeps_num * sweep_interval
    print(f"\n建议测试范围:")
    print(f"  最小帧索引: {min_frame} (需要 {sweeps_num} 个历史帧，间隔 {sweep_interval})")
    print(f"  最大帧索引: {len(data_infos) - 1}")
    print(f"  可用测试帧数: {len(data_infos) - min_frame}")
    
    print("\n✓ 验证完成！")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("自定义数据集元数据生成工具")
    print("=" * 60)
    
    # 检查目录
    print("\n检查数据目录...")
    assert os.path.exists(IMAGE_DIR), f"✗ 图像目录不存在: {IMAGE_DIR}"
    assert os.path.exists(RADAR_DIR), f"✗ Radar目录不存在: {RADAR_DIR}"
    assert os.path.exists(CALIB_DIR), f"✗ 标定目录不存在: {CALIB_DIR}"
    print("✓ 所有目录存在")
    
    # 生成数据信息
    data_infos = create_data_infos()
    
    # 保存
    save_data_infos(data_infos, OUTPUT_FILE)
    
    # 验证
    verify_data_infos(OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"输出文件: {OUTPUT_FILE}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
