"""
Script: Unzip SoccerNet Data
Description: Automatically extracts .zip files in the data directory.
"""
import os
import zipfile
from tqdm import tqdm

def unzip_files():
    # 1. 锁定数据目录
    # 路径: Tactix/data/SoccerNet/calibration-2023
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/SoccerNet/calibration-2023"))
    
    # 需要解压的文件列表
    zip_files = ["train.zip", "valid.zip", "test.zip", "challenge.zip"]
    
    print(f"📂 正在检查目录: {base_dir}")
    
    if not os.path.exists(base_dir):
        print("❌ 错误: 找不到数据目录，请检查下载是否成功。")
        return

    # 2. 遍历解压
    for z_file in zip_files:
        file_path = os.path.join(base_dir, z_file)
        
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过 {z_file} (文件不存在)")
            continue
            
        print(f"📦 正在解压: {z_file} ... (这可能需要几分钟)")
        
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # 获取压缩包内的文件列表，用于显示进度条
                members = zip_ref.infolist()
                
                # 使用 tqdm 显示解压进度
                for member in tqdm(members, desc=f"Extracting {z_file}", unit="file"):
                    zip_ref.extract(member, base_dir)
                    
            print(f"✅ {z_file} 解压完成！")
            
            # 可选：解压后删除压缩包以节省空间 (建议确认解压无误后再手动删)
            # os.remove(file_path) 
            
        except zipfile.BadZipFile:
            print(f"❌ 错误: {z_file} 似乎已损坏。")

    print("\n🎉 所有解压任务完成！现在可以运行 convert_to_yolo.py 了。")

if __name__ == "__main__":
    unzip_files()