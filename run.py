"""
Project: Tactix
File Created: 2026-02-04 16:14:41
Author: Xingnan Zhu
File Name: run.py
Description:
    The entry point for the Tactix application.
"""


import sys
import os

# 确保 tactix 包能被找到
sys.path.append(os.getcwd())

from tactix.engine.system import TactixEngine

if __name__ == "__main__":
    # 一键启动
    engine = TactixEngine()
    engine.run()