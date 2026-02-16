"""
Project: Tactix
File Created: 2026-02-14
Author: Xingnan Zhu
File Name: jersey_ocr.py
Description:
    Jersey number OCR detection using EasyOCR - enhances player identification.
    Supports jersey numbers 0-99.
    
    Enabled by default with graceful degradation (auto-disabled if easyocr not installed).
    
    Multi-ROI strategy: tries 3 vertical crop regions (top 30%, top 50%, full torso)
    to handle partial occlusions.
    
    Multi-variant preprocessing: original + high-contrast CLAHE + binary threshold
    to handle lighting variations.
    
    Two-digit recovery: handles two scenarios:
    1. "1X" → "X" (weight 0.3): Digit "1" easily mistaken from shadows/lines
       Example: true jersey "7", OCR sees "17" → "7" gets recovery vote
    2. "2X"-"9X" → "X" (weight 0.1): First digit occluded by arm/teammate (rare)
       Example: true jersey "27", first digit blocked → "7" gets weak recovery vote
    
    Returns the most confident valid result (1-2 digits, numeric only) via voting.
"""

from typing import Optional, List, Tuple
import numpy as np
import cv2


class JerseyOCR:
    """
    Multi-ROI, multi-variant OCR for jersey number detection.
    
    Uses EasyOCR with lazy initialization to avoid heavy startup cost
    when ENABLE_JERSEY_OCR is False.
    """
    
    def __init__(self, device: str = 'cpu', languages: List[str] = None, gpu: bool = None):
        """
        Initialize JerseyOCR (but don't load EasyOCR model yet).
        
        Args:
            device: 'cpu', 'cuda', or 'mps'
            languages: EasyOCR language codes (default: ['en'])
            gpu: Whether to use GPU for EasyOCR inference.
                 If None, auto-detected from device (True for 'cuda'/'mps').
        """
        self.languages = languages or ['en']
        self.gpu = gpu if gpu is not None else (device in ('cuda', 'mps'))
        self._reader = None  # Lazy-loaded on first detect() call
        
    def _ensure_reader(self) -> None:
        """Lazy-load EasyOCR reader on first use."""
        if self._reader is None:
            try:
                import easyocr
                self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
            except ImportError:
                raise ImportError(
                    "EasyOCR not installed. Run: pip install easyocr>=1.7.1"
                )
    
    def detect(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        min_h: int,
        min_w: int
    ) -> Optional[str]:
        """
        Detect jersey number from player bounding box.
        
        Args:
            frame: Full frame image (BGR)
            bbox: Player bounding box [x1, y1, x2, y2]
            min_h: Minimum bbox height to attempt OCR
            min_w: Minimum bbox width to attempt OCR
            
        Returns:
            Detected jersey number as string (e.g., "10", "7"), or None if no valid result
        """
        self._ensure_reader()
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        
        # Skip if bbox too small (distant player)
        if h < min_h or w < min_w:
            return None
        
        # Try multiple ROI crops (top 30%, top 50%, full torso)
        roi_configs = [
            (0.0, 0.3),   # Upper chest area
            (0.0, 0.5),   # Upper half
            (0.15, 0.6),  # Center torso (similar to shirt color extraction)
        ]
        
        all_results: List[Tuple[str, float]] = []
        
        for v_start, v_end in roi_configs:
            crop = self._extract_roi(frame, bbox, v_start, v_end)
            if crop is None:
                continue
            
            # Try multiple preprocessing variants
            variants = self._preprocess_variants(crop)
            
            for variant in variants:
                results = self._run_ocr(variant)
                all_results.extend(results)
        
        # Filter and select best result
        return self._select_best_result(all_results)
    
    def _extract_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        v_start: float,
        v_end: float
    ) -> Optional[np.ndarray]:
        """
        Extract vertical sub-region of player bounding box.
        
        Args:
            frame: Full frame image
            bbox: [x1, y1, x2, y2]
            v_start: Vertical start position (0.0 = top, 1.0 = bottom)
            v_end: Vertical end position
            
        Returns:
            Cropped ROI or None if invalid
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        
        crop_y1 = int(y1 + h * v_start)
        crop_y2 = int(y1 + h * v_end)
        
        # Horizontal crop: center 70% to avoid edge noise
        w = x2 - x1
        crop_x1 = int(x1 + w * 0.15)
        crop_x2 = int(x1 + w * 0.85)
        
        # Bounds check
        frame_h, frame_w = frame.shape[:2]
        if crop_y1 < 0 or crop_y2 > frame_h or crop_x1 < 0 or crop_x2 > frame_w:
            return None
        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None
        
        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if roi.size == 0:
            return None
        
        return roi
    
    def _preprocess_variants(self, crop: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessing variants of the crop.
        
        Args:
            crop: Input ROI image
            
        Returns:
            List of preprocessed variants (original, CLAHE, binary threshold)
        """
        variants = []
        
        # 1. Original
        variants.append(crop)
        
        # 2. High-contrast CLAHE (adaptive histogram equalization)
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(gray)
            variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass
        
        # 3. Binary threshold
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass
        
        return variants
    
    def _run_ocr(self, image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Run EasyOCR on preprocessed image.
        
        Args:
            image: Preprocessed ROI
            
        Returns:
            List of (text, confidence) tuples
        """
        try:
            results = self._reader.readtext(image)
            # EasyOCR returns list of ([bbox], text, conf)
            # We only need text and confidence
            return [(text, conf) for (_, text, conf) in results]
        except Exception:
            return []
    
    def _select_best_result(self, all_results: List[Tuple[str, float]]) -> Optional[str]:
        """
        Filter and select best jersey number (0-99) from all OCR results using voting strategy.
        
        Filtering rules:
        - Must be 1-2 digits (supports 0-99)
        - Must be numeric only
        - Confidence > 0.3
        
        Selection strategy:
        - Count votes for each number across all 9 OCR attempts (3 ROIs × 3 variants)
        - Apply two-digit recovery with differential weighting:
          * "1X" → "X" with 0.3 weight (digit "1" easily mistaken from shadows)
          * "2X"-"9X" → "X" with 0.1 weight (partial occlusion by arm/teammate)
        - Prefer two-digit if it has more direct votes than single digit
        - Select number with most votes; if tied, highest total confidence
        
        Example 1 (noise correction):
            OCR: ["17" (0.4), "7" (0.7), "7" (0.6), "7" (0.5), "17" (0.3)]
            Votes: "7": 3 direct + 2 recovery = 5, "17": 2
            Winner: "7" ✓
        
        Example 2 (two-digit preservation):
            OCR: ["27" (0.6), "27" (0.5), "27" (0.4), "7" (0.6)]
            Direct: "27": 3, "7": 1
            Winner: "27" ✓ (gets bonus for having more direct votes)
        
        Args:
            all_results: List of (text, confidence) tuples from all ROIs/variants
            
        Returns:
            Best valid jersey number or None
        """
        if not all_results:
            return None
        
        from collections import defaultdict
        
        # Vote counting: {number: [confidences]}
        votes: defaultdict = defaultdict(list)
        
        for text, conf in all_results:
            # Clean text (remove spaces, punctuation)
            cleaned = ''.join(c for c in text if c.isdigit())
            
            if not cleaned:
                continue
            
            # Must be 1-2 digits
            if len(cleaned) > 2:
                continue
            
            # Confidence threshold
            if conf < 0.3:
                continue
            
            # Vote for the detected number
            votes[cleaned].append(conf)
            
            # Two-digit recovery: handle partial occlusion or noise
            # Two scenarios:
            # 1. "1X" → "X": Digit "1" easily mistaken from shadows/lines (weight 0.3)
            # 2. "2X"-"9X" → "X": First digit occluded by arm/teammate (weight 0.1, rare)
            # Example: true jersey "7", OCR sees "17" → "7" gets recovery vote
            if len(cleaned) == 2:
                last_digit = cleaned[1]
                recovery_weight = 0.3 if cleaned[0] == '1' else 0.1
                votes[last_digit].append(conf * recovery_weight)
        
        if not votes:
            return None
        
        # Special case: if both "XY" and "Y" are candidates, prefer "XY" if it has strong direct votes
        # Handles conflict between two-digit number and its last digit
        # Example: "27" with 3 direct votes vs "7" with 1 direct + 3 recovery votes → choose "27"
        for two_digit in list(votes.keys()):
            if len(two_digit) == 2:
                last_digit = two_digit[1]
                if last_digit in votes:
                    # Count direct votes (confidence > 0.4 indicates direct detection, not recovery)
                    two_digit_direct = sum(1 for c in votes[two_digit] if c > 0.4)
                    single_digit_direct = sum(1 for c in votes[last_digit] if c > 0.4)
                    
                    # If two-digit has more direct votes, it's likely the true number
                    if two_digit_direct > single_digit_direct:
                        # Boost two-digit's total confidence to ensure it wins
                        votes[two_digit].append(0.5)
        
        # Select winner by vote count, then by total confidence
        winner = max(
            votes.items(),
            key=lambda x: (len(x[1]), sum(x[1]))  # (vote_count, total_confidence)
        )
        
        return winner[0]
