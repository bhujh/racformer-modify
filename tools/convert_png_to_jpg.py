#!/usr/bin/env python3
"""
将PNG图像转换为JPG格式

用法:
    python tools/convert_png_to_jpg.py
"""

import os
import cv2
from glob import glob
from tqdm import tqdm

DATA_ROOT = "/mnt/diskNvme0/wangpengData/OwnData/"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")

def convert_png_to_jpg():
    """将所有PNG文件转换为JPG"""
    print("\n" + "=" * 60)
    print("PNG到JPG转换工具")
    print("=" * 60)
    
    # 查找所有PNG文件
    png_files = sorted(glob(os.path.join(IMAGE_DIR, "*.png")))
    print(f"\n找到 {len(png_files)} 个PNG文件")
    
    if len(png_files) == 0:
        print("没有找到PNG文件！")
        return
    
    # 转换
    print("\n开始转换...")
    for png_file in tqdm(png_files):
        # 读取PNG
        img = cv2.imread(png_file)
        
        # 生成JPG文件名
        jpg_file = png_file.replace('.png', '.jpg')
        
        # 保存为JPG
        cv2.imwrite(jpg_file, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"\n✓ 转换完成！共转换 {len(png_files)} 个文件")
    print(f"✓ JPG文件保存在: {IMAGE_DIR}")
    
    # 询问是否删除PNG文件
    print("\n注意：PNG文件仍然保留。如果需要删除，请手动执行：")
    print(f"  rm {IMAGE_DIR}/*.png")

if __name__ == '__main__':
    convert_png_to_jpg()
