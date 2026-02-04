"""
Project: Tactix
File Created: 2026-02-02 16:16:34
Author: Xingnan Zhu
File Name: types.py
Description: xxx...
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

# ==========================================
# 1. 基础定义与枚举
# ==========================================

class TeamID(Enum):
    A = 0           # 主队
    B = 1           # 客队
    REFEREE = 2     # 裁判
    GOALKEEPER = 3  # 门将
    UNKNOWN = -1    # 未知

@dataclass
class Point:
    """代表一个坐标点，方便类型提示"""
    x: float
    y: float
    
    def to_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])

# ==========================================
# 2. 几何与骨骼 (Vision Layer Output)
# ==========================================

@dataclass
class Keypoints:
    """从 Pose 模型提取的关键点"""
    nose: Optional[np.ndarray] = None
    left_foot: Optional[np.ndarray] = None
    right_foot: Optional[np.ndarray] = None
    # 还可以加肩膀、膝盖等，如果需要分析跑姿
    
    @property
    def bottom_center(self) -> Optional[np.ndarray]:
        if self.left_foot is not None and self.right_foot is not None:
            return (self.left_foot + self.right_foot) / 2
        return self.left_foot or self.right_foot

# ==========================================
# 3. 核心实体 (Player & Ball)
# ==========================================

@dataclass
class Player:
    """
    球员实体：包含视觉信息 + 物理信息 + 战术状态
    """
    # --- 基础视觉信息 (由 Detector/Tracker 填充) ---
    id: int
    rect: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    class_id: int = 0
    confidence: float = 0.0
    team: TeamID = TeamID.UNKNOWN
    keypoints: Optional[Keypoints] = None

    # --- 物理信息 (由 Semantics Layer 填充) ---
    # 真实球场坐标 (单位: 米)。原点通常设为中圈或球场左上角。
    # 用于: 绘制俯视图、计算真实距离、Voronoi 图
    pitch_position: Optional[Point] = None 
    
    # 速度向量 (单位: 米/秒 或 像素/帧)
    # 用于: 显示速度条、预测跑位
    velocity: Optional[Point] = None 
    
    # 瞬时速度大小 (标量，方便直接显示 "30 km/h")
    speed: float = 0.0 
    
    # 身体朝向向量 (单位向量)
    # 用于: 传球概率计算 (人很难向背后传球)
    orientation: Optional[Point] = None

    # --- 历史信息 (用于绘制拖尾/轨迹) ---
    # 记录过去 N 帧的像素坐标 [(x,y), (x,y)...]
    trajectory: List[Point] = field(default_factory=list)

    @property
    def anchor(self) -> Tuple[int, int]:
        """绘图锚点 (优先取脚下)"""
        if self.keypoints and self.keypoints.bottom_center is not None:
            return tuple(self.keypoints.bottom_center.astype(int))
        x1, y1, x2, y2 = self.rect
        return int((x1 + x2) / 2), int(y2)

@dataclass
class Ball:
    rect: Tuple[float, float, float, float]
    score: float = 0.0
    pitch_position: Optional[Point] = None # 球的真实位置
    velocity: Optional[Point] = None       # 球速 (用于判断射门/传球力度)
    owner_id: Optional[int] = None         # 持球人 ID
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.rect
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

# ==========================================
# 4. 战术分析结果 (Analysis Layer Output)
# ==========================================

@dataclass
class PassEvent:
    """定义一个传球事件 (用于传球网络)"""
    sender_id: int
    receiver_id: int
    start_frame: int
    end_frame: int
    is_successful: bool = True

@dataclass
class FrameAnalysis:
    """
    存储当前帧的高级战术元数据
    """
    # --- 越位线 ---
    # 最后的后卫在球场坐标系中的 X 坐标 (假设 X 轴是底线方向)
    offside_line_x: Optional[float] = None 
    # 投影回屏幕上的越位线坐标点 [(x1,y1), (x2,y2)]
    offside_line_pixels: Optional[Tuple[Point, Point]] = None
    
    # --- 空间控制 (Voronoi/Pitch Control) ---
    # 简单的: 哪队控制了更多区域 (0.0 - 1.0)
    possession_value: float = 0.5 
    
    # --- 传球网络 ---
    # 当前持球人的潜在接球点: {teammate_id: probability/score}
    pass_candidates: Dict[int, float] = field(default_factory=dict)

# ==========================================
# 5. 帧总线 (The Bus)
# ==========================================

@dataclass
class FrameData:
    frame_index: int
    image_shape: Tuple[int, int]
    
    # 实体
    players: List[Player] = field(default_factory=list)
    ball: Optional[Ball] = None
    
    # 相机数据 (用于 AR 稳定)
    # 单应性矩阵 (3x3 matrix)，负责 Pixel <-> Meter 转换
    homography: Optional[np.ndarray] = None 
    
    # 战术数据
    analysis: Optional[FrameAnalysis] = None
    
    def get_team_players(self, team: TeamID) -> List[Player]:
        return [p for p in self.players if p.team == team]

    def get_player_by_id(self, pid: int) -> Optional[Player]:
        for p in self.players:
            if p.id == pid: return p
        return None
    
# ==========================================
# 6. 球场标准 (FIFA / UEFA Standard)
# ==========================================
class PitchConfig:
    # 真实世界的逻辑尺寸 (单位: 米)
    # 这将用于计算真实速度和距离
    LENGTH = 105.0
    WIDTH = 68.0
    
    # 注意：生成图片后，请你去 assets/pitch_bg.png 查看一下它的详细信息
    # Mac 上右键图片 -> 显示简介 -> 尺寸
    PIXEL_WIDTH = 1501
    PIXEL_HEIGHT = 1010
    
    # 自动计算比例尺 (像素/米)
    X_SCALE = PIXEL_WIDTH / LENGTH
    Y_SCALE = PIXEL_HEIGHT / WIDTH