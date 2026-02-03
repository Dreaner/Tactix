"""
Project: Tactix
File Created: 2026-02-02 23:11:14
Author: Xingnan Zhu
File Name: generate_pitch.py
Description: xxx...
"""


import os
import matplotlib.pyplot as plt
from mplsoccer import Pitch

def generate_tactix_pitch():
    # 1. 确保输出目录存在
    # 假设我们想存到项目根目录下的 assets/ 文件夹
    # 获取当前脚本的绝对路径，然后往上找两层
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir)) # 回到 Tactix/
    assets_dir = os.path.join(project_root, "assets")
    
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
        print(f"📁 Created assets directory: {assets_dir}")

    # 2. 初始化场地 (FIFA/UEFA 标准: 105m x 68m)
    # pitch_type='uefa': 自动设定为 105x68 米
    # axis=True, label=True: 先开启坐标轴，方便你看清楚尺寸，正式生成时可以关掉
    pitch = Pitch(
        pitch_type='uefa',      
        pitch_color='grass',  # 深墨绿色底
        line_color="#ffffff",   # 浅灰白色线
        stripe=True,
        linewidth=2,
        pad_left=0, 
        pad_right=0,
        pad_bottom=0,
        pad_top=0
    )
    
    # 3. 画图
    # figsize 控制清晰度，(16, 10.4) 刚好对应 105:68 的比例
    fig, ax = pitch.draw(figsize=(16, 10.4))
    
    # 4. 保存为素材
    output_path = os.path.join(assets_dir, "pitch_bg.png")
    
    # 保存时去掉白边 (bbox_inches='tight', pad_inches=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close() # 释放内存
    
    print(f"✅ 标准战术板 (105x68m) 已生成: {output_path}")

if __name__ == "__main__":
    generate_tactix_pitch()