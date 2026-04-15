#!/usr/bin/env python3
"""
分析检测结果

用法:
    python tools/analyze_results.py --results results/own_data_results.pkl
"""

import pickle
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='分析检测结果')
    parser.add_argument('--results', required=True, help='检测结果pkl文件')
    parser.add_argument('--score_thr', type=float, default=0.3, help='置信度阈值')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("检测结果分析")
    print("="*60)
    
    # 加载结果
    print(f"\n加载结果: {args.results}")
    with open(args.results, 'rb') as f:
        results = pickle.load(f)
    
    print(f"✓ 共 {len(results)} 帧")
    
    # 类别名称
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]
    
    # 统计信息
    total_detections = 0
    class_counts = {}
    score_list = []
    frames_with_detections = 0
    
    for result in results:
        if 'boxes_3d' not in result:
            continue
        
        num_boxes = len(result['boxes_3d'])
        if num_boxes > 0:
            frames_with_detections += 1
            total_detections += num_boxes
            
            if 'scores_3d' in result:
                scores = result['scores_3d'].cpu().numpy()
                score_list.extend(scores.tolist())
            
            if 'labels_3d' in result:
                labels = result['labels_3d'].cpu().numpy()
                for label in labels:
                    class_counts[int(label)] = class_counts.get(int(label), 0) + 1
    
    # 输出统计
    print(f"\n{'='*60}")
    print("总体统计")
    print(f"{'='*60}")
    print(f"总帧数: {len(results)}")
    print(f"有检测结果的帧数: {frames_with_detections}")
    print(f"检测到的目标总数: {total_detections}")
    print(f"平均每帧检测数: {total_detections / len(results):.2f}")
    
    if score_list:
        scores_array = np.array(score_list)
        print(f"\n置信度统计:")
        print(f"  最小值: {scores_array.min():.4f}")
        print(f"  最大值: {scores_array.max():.4f}")
        print(f"  平均值: {scores_array.mean():.4f}")
        print(f"  中位数: {np.median(scores_array):.4f}")
        
        # 按阈值统计
        for thr in [0.1, 0.3, 0.5, 0.7]:
            count = (scores_array > thr).sum()
            print(f"  > {thr}: {count} 个 ({count/len(scores_array)*100:.1f}%)")
    
    if class_counts:
        print(f"\n{'='*60}")
        print("各类别统计")
        print(f"{'='*60}")
        for class_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            percentage = count / total_detections * 100
            print(f"  {class_name:20s}: {count:4d} 个 ({percentage:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("✓ 分析完成！")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
