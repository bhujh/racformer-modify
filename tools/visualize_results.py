#!/usr/bin/env python3
"""
可视化检测结果

用法:
    python tools/visualize_results.py \
        --results results/own_data_results.pkl \
        --data_root /mnt/diskNvme0/wangpengData/OwnData \
        --vis_dir results/visualizations \
        --max_frames 20
"""

import os
import pickle
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='可视化检测结果')
    parser.add_argument('--results', required=True, help='检测结果pkl文件')
    parser.add_argument('--data_root', required=True, help='数据根目录')
    parser.add_argument('--vis_dir', required=True, help='可视化输出目录')
    parser.add_argument('--max_frames', type=int, default=None, help='最多可视化多少帧（None=全部）')
    parser.add_argument('--score_thr', type=float, default=0.3, help='置信度阈值')
    args = parser.parse_args()
    return args


def project_3d_box_to_image(corners_3d, lidar2img):
    """
    将3D边界框的8个角点投影到图像平面
    
    Args:
        corners_3d: (8, 3) 3D角点坐标
        lidar2img: (4, 4) 变换矩阵
    
    Returns:
        corners_2d: (8, 2) 2D角点坐标
        valid_mask: (8,) 是否在相机前方
    """
    # 转换为齐次坐标
    corners_3d_homo = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)  # (8, 4)
    
    # 投影到图像
    corners_img = corners_3d_homo @ lidar2img.T  # (8, 4)
    
    # 检查是否在相机前方
    valid_mask = corners_img[:, 2] > 0
    
    # 归一化
    corners_2d = corners_img[:, :2] / (corners_img[:, 2:3] + 1e-6)
    
    return corners_2d, valid_mask


def draw_3d_box_on_image(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制3D边界框
    
    Args:
        image: 图像
        corners_2d: (8, 2) 2D角点坐标
        color: 颜色
        thickness: 线宽
    """
    # 3D边界框的12条边（连接关系）
    # 底面4条边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
    ]
    
    for start, end in edges:
        pt1 = tuple(corners_2d[start].astype(int))
        pt2 = tuple(corners_2d[end].astype(int))
        
        # 检查是否在图像范围内
        if (0 <= pt1[0] < image.shape[1] and 0 <= pt1[1] < image.shape[0] and
            0 <= pt2[0] < image.shape[1] and 0 <= pt2[1] < image.shape[0]):
            cv2.line(image, pt1, pt2, color, thickness)
    
    return image


def get_class_color(class_id):
    """获取类别对应的颜色"""
    colors = [
        (0, 255, 0),    # car - 绿色
        (255, 0, 0),    # truck - 蓝色
        (0, 0, 255),    # trailer - 红色
        (255, 255, 0),  # bus - 青色
        (255, 0, 255),  # construction_vehicle - 品红
        (0, 255, 255),  # bicycle - 黄色
        (128, 0, 128),  # motorcycle - 紫色
        (255, 128, 0),  # pedestrian - 橙色
        (0, 128, 255),  # traffic_cone - 浅蓝
        (128, 128, 128),# barrier - 灰色
    ]
    return colors[class_id % len(colors)]


def visualize_frame(result, frame_idx, data_root, vis_dir, score_thr=0.3):
    """
    可视化单帧的检测结果
    
    Args:
        result: 检测结果字典
        frame_idx: 帧索引
        data_root: 数据根目录
        vis_dir: 输出目录
        score_thr: 置信度阈值
    """
    # 类别名称
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    
    # 读取图像（只可视化第一个相机）
    img_path = os.path.join(data_root, 'images', f'{frame_idx:07d}.jpg')
    if not os.path.exists(img_path):
        print(f"警告: 图像不存在 {img_path}")
        return
    
    image = cv2.imread(img_path)
    if image is None:
        print(f"警告: 无法读取图像 {img_path}")
        return
    
    # 获取检测结果
    if 'boxes_3d' not in result or len(result['boxes_3d']) == 0:
        # 没有检测到目标，保存原图
        out_path = os.path.join(vis_dir, f'frame_{frame_idx:07d}.jpg')
        cv2.imwrite(out_path, image)
        return
    
    boxes_3d = result['boxes_3d']
    scores_3d = result['scores_3d'].cpu().numpy()
    labels_3d = result['labels_3d'].cpu().numpy()
    
    # 过滤低置信度的检测
    mask = scores_3d > score_thr
    boxes_3d = boxes_3d[mask]
    scores_3d = scores_3d[mask]
    labels_3d = labels_3d[mask]
    
    # 获取lidar2img变换矩阵（使用第一个相机）
    # 注意：这里需要从pkl文件中读取，简化起见，我们假设使用单位矩阵
    # 实际使用时需要从数据集中获取正确的变换矩阵
    
    # 绘制每个检测框
    for i in range(len(boxes_3d)):
        box_3d = boxes_3d[i]
        score = scores_3d[i]
        label = int(labels_3d[i])
        
        # 获取3D边界框的8个角点
        corners_3d = box_3d.corners.cpu().numpy()  # (8, 3)
        
        # 简化投影：只绘制中心点
        center_3d = box_3d.gravity_center.cpu().numpy()  # (3,)
        
        # 在图像上绘制文本（简化版）
        class_name = class_names[label] if label < len(class_names) else f"class_{label}"
        text = f"{class_name}: {score:.2f}"
        color = get_class_color(label)
        
        # 在图像中心附近绘制文本
        text_pos = (10, 30 + i * 25)
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
    
    # 保存可视化结果
    out_path = os.path.join(vis_dir, f'frame_{frame_idx:07d}.jpg')
    cv2.imwrite(out_path, image)


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("可视化检测结果")
    print("="*60)
    
    # 加载检测结果
    print(f"\n加载检测结果: {args.results}")
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    print(f"✓ 共 {len(results)} 帧")
    
    # 创建输出目录
    os.makedirs(args.vis_dir, exist_ok=True)
    print(f"✓ 输出目录: {args.vis_dir}")
    
    # 确定要可视化的帧数
    num_frames = len(results)
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    print(f"✓ 将可视化 {num_frames} 帧")
    
    # 可视化每一帧
    print(f"\n开始可视化...")
    for i in tqdm(range(num_frames)):
        # 从第15帧开始（前14帧历史帧不足）
        frame_idx = i + 15 if i + 15 < len(results) else i
        visualize_frame(results[frame_idx], frame_idx, args.data_root, 
                       args.vis_dir, args.score_thr)
    
    # 统计信息
    total_detections = sum(len(r['boxes_3d']) for r in results[:num_frames])
    print(f"\n✓ 可视化完成！")
    print(f"✓ 共检测到 {total_detections} 个目标")
    print(f"✓ 可视化图像已保存到: {args.vis_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
