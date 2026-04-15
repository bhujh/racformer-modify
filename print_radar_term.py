import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud

# 1. 初始化 nuScenes 对象
nusc = NuScenes(version="v1.0-mini", dataroot="/mnt/diskNvme1/dataset/nuscenes_mini", verbose=True)

# 2. 获取第一个样本（sample）
my_sample = nusc.sample[0]

# 3. 获取前向雷达（RADAR_FRONT）的 sample_data token
radar_token = my_sample["data"]["RADAR_FRONT"]

# 4. 获取该雷达数据的完整信息字典
radar_sd = nusc.get("sample_data", radar_token)

# 5. 打印 sample_data 字段信息
print("=== 雷达数据 (sample_data) 字段 ===")
for key, value in radar_sd.items():
    print(f"{key}: {value}")

# 6. 获取雷达点云文件的完整路径
radar_file_path = os.path.join(nusc.dataroot, radar_sd["filename"])
print(f"\n=== 雷达点云文件路径 ===")
print(f"文件路径: {radar_file_path}")

# 7. 加载雷达点云数据
radar_pc = RadarPointCloud.from_file(radar_file_path)

# 8. 打印点云的形状和字段信息
print(f"\n=== 雷达点云数据形状 ===")
print(f"点云形状: {radar_pc.points.shape}")
print(f"字段数量: {radar_pc.points.shape[0]}")
print(f"点的数量: {radar_pc.points.shape[1]}")

# 9. 输出前5个点的所有字段信息
print(f"\n=== 前5个点的所有字段信息 ===")
num_points_to_show = min(15, radar_pc.points.shape[1])

# RadarPointCloud 的字段通常包括：
# 0: x (米)
# 1: y (米)
# 2: z (米)
# 3: dyn_prop (动态属性)
# 4: id (ID)
# 5: rcs (雷达散射截面)
# 6: vx (x方向速度)
# 7: vy (y方向速度)
# 8: vx_comp (补偿后的x方向速度)
# 9: vy_comp (补偿后的y方向速度)
# 10: is_quality_valid (质量是否有效)
# 11: ambig_state (模糊状态)
# 12: x_rms (x的均方根误差)
# 13: y_rms (y的均方根误差)
# 14: invalid_state (无效状态)
# 15: pdh0 (检测假设0的概率密度)
# 16: vx_rms (vx的均方根误差)
# 17: vy_rms (vy的均方根误差)

field_names = ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 
               'vy_comp', 'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms', 
               'invalid_state', 'pdh0', 'vx_rms', 'vy_rms']

for i in range(num_points_to_show):
    print(f"\n--- 点 {i+1} ---")
    for j, field_name in enumerate(field_names[:radar_pc.points.shape[0]]):
        print(f"  {field_name}: {radar_pc.points[j, i]}")
