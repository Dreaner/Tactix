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
    PITCH_MODEL_PATH: str = "assets/weights/football_pitch.pt"
    # 你的球员模型 (Model B)
    PLAYER_MODEL_PATH: str = "assets/weights/football_v1.pt"
    
    INPUT_VIDEO: str = "assets/samples/InterGoalClip.mp4"
    OUTPUT_VIDEO: str = "assets/output/Final_V4_Result.mp4"
    PITCH_TEMPLATE: str = "assets/pitch_bg.png"

    # === 模型参数 ===
    DEVICE: str = "mps"  # Mac用 'mps', Windows用 'cuda', 只有CPU用 'cpu'
    
    # 置信度阈值
    CONF_PITCH: float = 0.3  # 球场关键点要准一点
    CONF_PLAYER: float = 0.3 # 球员检测稍微宽容点

    # === 战术参数 ===
    MAX_PASS_DIST: int = 400
    BALL_OWNER_DIST: int = 60


    # === [新增] 调试模式开关 ===
    USE_MOCK_PITCH: bool = False  # <--- 打开这个！
    
    # 这是 InterGoalClip.mp4 的手动校准数据 (你之前 main.py 里的)
    # 格式: [x, y, keypoint_id]
    # ID 来自 keypoints.py 的 YOLO_INDEX_MAP:
    # 9=L_PA_TOP_LINE, 3=MID_TOP, 2=CIRCLE_BOTTOM, 15=L_PENALTY_SPOT
    MOCK_KEYPOINTS = [
        (137, 89, 9),    # L_PA_TOP_LINE
        (1126, 87, 3),   # MID_TOP
        (1045, 398, 2),  # CIRCLE_BOTTOM
        (138, 222, 15)   # L_PENALTY_SPOT
    ]