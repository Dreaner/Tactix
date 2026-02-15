"""
Tactix — Automated Football Tactical Analysis Engine
=====================================================
Quick start::

    from tactix import TactixEngine, Config

    cfg = Config()
    cfg.INPUT_VIDEO  = "match.mp4"
    cfg.OUTPUT_VIDEO = "match_out.mp4"
    cfg.SHOW_MINIMAP = True

    engine = TactixEngine(cfg)
    engine.run()
"""
from tactix.config import CalibrationMode, Config
from tactix.engine.system import TactixEngine

__version__ = "0.1.0"
__all__ = ["TactixEngine", "Config", "CalibrationMode", "__version__"]
