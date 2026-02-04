"""
Project: Tactix
File Created: 2026-02-02 11:56:08
Author: Xingnan Zhu
File Name: config.py
Description: 把所有路径、颜色、参数都放在这里。
"""


from dataclasses import dataclass

@dataclass
class Config:
    # === 路径设置 ===
    # 你的球场模型 (Model A)
    PITCH_MODEL_PATH: str = "assets/weights/best.pt"
    # 你的球员模型 (Model B)
    PLAYER_MODEL_PATH: str = "assets/weights/yolov8m.pt"
    
    INPUT_VIDEO: str = "assets/samples/InterGoalClip.mp4"
    OUTPUT_VIDEO: str = "assets/output/Final_V4_Result.mp4"
    PITCH_TEMPLATE: str = "assets/pitch_bg.png"

    # === 模型参数 ===
    DEVICE: str = "mps"  # Mac用 'mps', Windows用 'cuda', 只有CPU用 'cpu'
    
    # 置信度阈值
    CONF_PITCH: float = 0.5  # 球场关键点要准一点
    CONF_PLAYER: float = 0.3 # 球员检测稍微宽容点

    # === 战术参数 ===
    MAX_PASS_DIST: int = 400
    BALL_OWNER_DIST: int = 60