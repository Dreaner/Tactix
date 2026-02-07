"""
Project: Tactix
File Created: 2026-02-05 18:37:28
Author: Xingnan Zhu
File Name: base.py
Description:
    Defines the abstract base class for all datasets exporters.
    Ensures a consistent interface for exporting frame datasets to various formats (JSON, CSV, etc.).
"""

from abc import ABC, abstractmethod
from tactix.core.types import FrameData

class BaseExporter(ABC):
    """
    Abstract base class for datasets export.
    """
    
    @abstractmethod
    def add_frame(self, frame_data: FrameData):
        """
        Process a single frame of datasets and buffer it for export.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Finalize and write the datasets to the output destination (file, DB, etc.).
        """
        pass
