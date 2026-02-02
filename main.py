"""
Project: Tactix
File Created: 2026-02-02 11:55:51
Author: Xingnan Zhu
File Name: main.py
Description: xxx...
"""


from tactix.core.engine import Engine
from tactix.config import Config

if __name__ == "__main__":
    print(f"Tactix running on: {Config.DEVICE}")
    # engine = Engine("assets/match.mp4")
    # engine.run()