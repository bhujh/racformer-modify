"""
自定义数据集类
用于加载单相机+单radar的数据
"""

import os
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from pyquaternion import Quaternion


@DATASETS.register_module()
class OwnDataset(Custom3DDataset):
    """
    自定义数据集，用于加载单相机+单radar的数据
    
    关键特性：
    1. 单相机复制到6个位置，模拟多相机配置
    2. 可配置的历史帧数量和帧间隔
    3. 支持.pcd和.npy格式的radar数据
    
    Args:
        sweeps_num: 历史帧数量（默认7）
        sweep_interval: 帧间隔（默认2，适用于15Hz数据）
        **kwargs: 传递给父类的其他参数
    """
    
    def __init__(self, 
                 sweeps_num=7,           # 历史帧数量
                 sweep_interval=2,       # 帧间隔
                 **kwargs):
        """初始化数据集"""
        self.sweeps_num = sweeps_num
        self.sweep_interval = sweep_interval
        super().__init__(**kwargs)
        
        print(f"\n{'='*60}")
        print(f"OwnDataset 初始化完成")
        print(f"{'='*60}")
        print(f"  数据集大小: {len(self.data_infos)}")
        print(f"  历史帧数量: {self.sweeps_num}")
        print(f"  帧间隔: {self.sweep_interval}")
        print(f"  需要历史帧数: {self.sweeps_num * self.sweep_interval}")
        print(f"  建议测试范围: 第{self.sweeps_num * self.sweep_interval}帧到第{len(self.data_infos)}帧")
        print(f"  可用测试帧数: {len(self.data_infos) - self.sweeps_num * self.sweep_interval}")
        print(f"{'='*60}\n")
    
    def collect_sweeps(self, index, into_past=60):
        """
        采样历史帧
        
        对于102帧数据，sweep_interval=2，sweeps_num=7：
        - 需要向前查找 7*2=14 帧
        - 因此前14帧无法获得完整历史帧
        - 建议从第15帧开始测试
        
        Args:
            index: 当前帧索引
            into_past: 最多向前查找的帧数（未使用，保持接口一致）
        
        Returns:
            all_sweeps_prev: 历史帧列表
            all_sweeps_next: 未来帧列表（空列表）
        """
        all_sweeps_prev = []
        
        for k in range(self.sweeps_num):
            # 计算历史帧索引
            sweep_idx = index - (k + 1) * self.sweep_interval
            
            # 边界处理：如果超出范围，复制第0帧
            if sweep_idx < 0:
                sweep_idx = 0
            
            sweep_info = self.data_infos[sweep_idx]
            
            # 转换路径为绝对路径
            sweep_dict = {}
            for cam_key, cam_info in sweep_info['cams'].items():
                cam_info_abs = cam_info.copy()
                cam_info_abs['data_path'] = os.path.join(self.data_root, cam_info['data_path'])
                sweep_dict[cam_key] = cam_info_abs
            
            for rad_key, rad_info in sweep_info['rads'].items():
                rad_info_abs = rad_info.copy()
                rad_info_abs['data_path'] = os.path.join(self.data_root, rad_info['data_path'])
                sweep_dict[rad_key] = rad_info_abs
            
            all_sweeps_prev.append(sweep_dict)
        
        # 不使用未来帧
        all_sweeps_next = []
        
        return all_sweeps_prev, all_sweeps_next
    
    def get_data_info(self, index):
        """
        获取单帧数据信息
        
        关键：将单相机信息复制到6个位置
        
        Args:
            index: 帧索引
        
        Returns:
            input_dict: 包含所有必要信息的字典
        """
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)
        
        # 坐标系变换信息
        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        
        # 转换四元数为旋转矩阵
        ego2global_rotation = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation = Quaternion(lidar2ego_rotation).rotation_matrix
        
        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            pts_filename=os.path.join(self.data_root, info['lidar_path']),  # 使用绝对路径
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation,
        )
        
        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            cam_intrinsics = []
            
            # 获取单相机信息
            cam_info = info['cams']['CAM_FRONT']
            
            # 复制到6个位置（模拟多相机配置）
            for _ in range(6):
                # 使用绝对路径
                img_path = os.path.join(self.data_root, cam_info['data_path'])
                img_paths.append(img_path)
                img_timestamps.append(cam_info['timestamp'] / 1e6)
                
                # 计算lidar2img变换
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
            
            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                intrinsics=cam_intrinsics,
            ))
        
        # 测试模式不需要标注
        if not self.test_mode:
            # 如果有标注数据，可以在这里添加
            pass
        
        return input_dict
