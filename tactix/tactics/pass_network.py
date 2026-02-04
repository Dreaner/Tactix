"""
Project: Tactix
File Created: 2026-02-02 16:34:42
Author: Xingnan Zhu
File Name: pass_network.py
Description: xxx...
"""

# 文件路径: tactix/tactix/tactics/pass_network.py

from typing import List, Tuple, Optional

import numpy as np

from tactix.core.types import FrameData, Player


class PassNetwork:
    def __init__(self, max_pass_dist=300, ball_owner_dist=50):
        self.max_pass_dist = max_pass_dist # 超过这个像素距离就不画线了
        self.ball_owner_dist = ball_owner_dist # 球离人多近算“控球”
    
    def analyze(self, frame_data: FrameData) -> List[Tuple[Tuple[int,int], Tuple[int,int], float]]:
        """
        返回需要绘制的连线列表: [(start_xy, end_xy, opacity), ...]
        """
        if not frame_data.ball or not frame_data.players:
            return []

        ball_center = np.array(frame_data.ball.center)
        # === 调试打印 1 ===
        print(f"Ball detected at {ball_center}")

        owner: Optional[Player] = None
        min_dist = float('inf')

        for p in frame_data.players:
            # 必须离球足够近，且我们知道他是哪队的
            # if p.team == TeamID.UNKNOWN or p.team == TeamID.REFEREE:
            #     continue
                
            # 使用脚下锚点计算距离更准
            dist = np.linalg.norm(np.array(p.anchor) - ball_center)
            if dist < min_dist:
                min_dist = dist
                owner = p

        # === 调试打印 2 ===
        print(f"Nearest player dist: {min_dist:.1f} px")
        
        # 记录持球人ID到球对象中
        frame_data.ball.owner_id = owner.id
        
        # 放宽距离限制！从 50 改到 100 甚至 150，先看看能不能画出来
        # self.ball_owner_dist 可以在 main.py 里传参，也可以这里写死测试
        effective_limit = max(self.ball_owner_dist, 100) 
        
        if min_dist > effective_limit:
            return []
            
        # === 调试打印 3 ===
        print(f"✅ Owner Found! ID: {owner.id}, Team: {owner.team}")

        # 2. 计算传球路线
        # 找出所有同队队友
        teammates = [p for p in frame_data.players if p.team == owner.team and p.id != owner.id]
        
        lines_to_draw = []
        
        for mate in teammates:
            # 计算距离
            dist = np.linalg.norm(np.array(owner.anchor) - np.array(mate.anchor))
            
            # 只有在射程范围内的才画线
            if dist < self.max_pass_dist:
                # 距离越近，线越亮 (opacity 越高)
                opacity = 1.0 - (dist / self.max_pass_dist)
                # 设定最小透明度，别太淡了
                opacity = max(0.2, opacity)
                
                lines_to_draw.append((owner.anchor, mate.anchor, opacity))
                
        return lines_to_draw
