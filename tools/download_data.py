"""
Script: Download SoccerNet Calibration Data
Run this from the project root: python3 tools/download_data.py
"""
import os
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

def main():
    # 1. 设置下载路径
    # 我们希望数据存在项目根目录下的 'datasets/SoccerNet' 文件夹里
    # os.path.dirname(...) 获取当前脚本所在目录 (tools/)
    # os.path.abspath(...) 转为绝对路径
    # ../datasets/SoccerNet  跳到上一级(根目录)的 datasets 文件夹
    local_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../datasets/SoccerNet"))
    
    print(f"📂 数据将下载到: {local_directory}")
    
    # 2. 初始化下载器
    my_downloader = SNdl(LocalDirectory=local_directory)

    # 3. 开始下载
    # ⚠️ 注意：完整数据集非常大！
    # 如果只是想测试流程，可以只下载 "challenge" 或 "test"
    # 如果要训练，通常需要 "train" 和 "valid"
    print("🚀 开始下载 SoccerNet Calibration 数据...")
    
    my_downloader.downloadDataTask(
        task="calibration-2023", 
        split=["train", "valid", "test", "challenge"] # 根据你的硬盘空间决定要不要全下
    )

    print("\n✅ 下载任务完成！")

if __name__ == "__main__":
    main()