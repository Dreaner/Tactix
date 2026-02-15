"""
Project: Tactix
File Created: 2026-02-15 12:00:00
Author: Xingnan Zhu
File Name: hybrid_estimator.py
Description:
    Implements a Hybrid pitch estimator that fuses AI (YOLO) keypoint detection
    with ORB-based global camera motion tracking.

    Strategy per frame:
      1. Run YOLO to detect pitch keypoints.
      2. If YOLO returns ≥ ANCHOR_THRESHOLD high-confidence points:
         → Use YOLO results directly and RESET the ORB tracker (eliminate drift).
      3. If YOLO returns 1–3 points:
         → Project previous keypoints via ORB motion, then blend with YOLO points.
      4. If YOLO returns 0 points:
         → Use ORB-only projection with decaying confidence.

    This ensures robust calibration even during fast camera pans when YOLO
    temporarily loses most keypoints.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from tactix.vision.calibration.base import BasePitchEstimator
from tactix.vision.calibration.ai_estimator import AIPitchEstimator
from tactix.core.keypoints import YOLO_INDEX_MAP


class HybridPitchEstimator(BasePitchEstimator):
    """
    Fuses YOLO keypoint detection with ORB-based inter-frame motion tracking.
    Provides continuous keypoint output even when YOLO detection degrades.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "mps",
        orb_features: int = 1000,
        max_drift_frames: int = 30,
        anchor_threshold: int = 4,
        conf_pitch: float = 0.3,
    ) -> None:
        print("🔀 Initializing Hybrid Pitch Estimator (AI + ORB)...")
        self.ai_estimator = AIPitchEstimator(model_path, device)
        self.orb_features = orb_features
        self.max_drift_frames = max_drift_frames
        self.anchor_threshold = anchor_threshold
        self.conf_pitch = conf_pitch

        # ORB motion-tracking state
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_keypoints: Optional[np.ndarray] = None  # (27, 2) — last known positions
        self._prev_confs: Optional[np.ndarray] = None       # (27,)
        self._drift_count: int = 0  # consecutive frames without a YOLO anchor reset
        self._calibration_source: str = "NONE"  # for debug overlay

    @property
    def calibration_source(self) -> str:
        """Returns the source label of the last calibration: 'AI', 'BLEND', 'ORB', 'NONE'."""
        return self._calibration_source

    def predict(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Predict 27 pitch keypoints for the given frame.

        Returns:
            (keypoints_xy (27,2), confidences (27,))  or  (None, None) on total failure.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Step 1: Run YOLO ---
        yolo_xy, yolo_conf = self.ai_estimator.predict(frame)

        # Count high-confidence YOLO detections
        n_yolo = 0
        if yolo_xy is not None and yolo_conf is not None:
            n_yolo = int(np.sum(yolo_conf >= self.conf_pitch))

        # --- Step 2: Compute ORB inter-frame motion (if we have a previous frame) ---
        M = self._estimate_motion(curr_gray)

        # --- Step 3: Fusion logic ---
        if n_yolo >= self.anchor_threshold:
            # YOLO has enough points — trust it fully, reset ORB tracker
            out_xy = yolo_xy.copy()
            out_conf = yolo_conf.copy()
            self._drift_count = 0
            self._calibration_source = "AI"

        elif self._prev_keypoints is not None and M is not None:
            # We can project previous keypoints via ORB motion
            projected_xy = self._apply_motion(self._prev_keypoints, M)

            if n_yolo > 0:
                # Partial YOLO — blend: use YOLO points where confident, ORB elsewhere
                out_xy, out_conf = self._blend(
                    yolo_xy, yolo_conf, projected_xy, self._prev_confs
                )
                self._drift_count += 1
                self._calibration_source = "BLEND"
            else:
                # No YOLO at all — pure ORB projection with decaying confidence
                self._drift_count += 1
                decay = max(0.0, 1.0 - self._drift_count / self.max_drift_frames)
                out_xy = projected_xy
                out_conf = self._prev_confs * decay
                self._calibration_source = "ORB"

        elif n_yolo > 0:
            # First frame or no motion estimate, but YOLO gave *some* points
            out_xy = yolo_xy.copy()
            out_conf = yolo_conf.copy()
            self._drift_count = 0
            self._calibration_source = "AI"
        else:
            # Total failure: no YOLO, no previous state
            self._prev_gray = curr_gray
            self._calibration_source = "NONE"
            return None, None

        # --- Step 4: Update state for next frame ---
        self._prev_gray = curr_gray
        self._prev_keypoints = out_xy.copy()
        self._prev_confs = out_conf.copy()

        return out_xy, out_conf

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_motion(self, curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate the inter-frame homography M (prev → curr) using ORB + RANSAC.
        Returns 3×3 matrix or None on failure.
        """
        if self._prev_gray is None:
            return None

        orb = cv2.ORB_create(nfeatures=self.orb_features)
        kp1, des1 = orb.detectAndCompute(self._prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        # Keep top 50% of matches
        good = matches[: max(4, len(matches) // 2)]
        if len(good) < 4:
            return None

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is None or mask is None:
            return None

        inliers = int(np.sum(mask))
        if inliers < 4:
            return None

        return M

    def _apply_motion(self, keypoints: np.ndarray, M: np.ndarray) -> np.ndarray:
        """
        Project (27,2) keypoints from the previous frame to the current frame via M.
        """
        # perspectiveTransform expects (N,1,2), float32
        pts = keypoints.astype(np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts, M)
        return projected.reshape(-1, 2)

    def _blend(
        self,
        yolo_xy: np.ndarray,
        yolo_conf: np.ndarray,
        orb_xy: np.ndarray,
        orb_conf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each of 27 keypoints:
          - If YOLO is confident (≥ conf_pitch) → use YOLO
          - Otherwise → use ORB projection (with decayed confidence)
        """
        out_xy = orb_xy.copy()
        out_conf = orb_conf.copy() * max(0.0, 1.0 - self._drift_count / self.max_drift_frames)

        for i in range(len(yolo_conf)):
            if yolo_conf[i] >= self.conf_pitch:
                out_xy[i] = yolo_xy[i]
                out_conf[i] = yolo_conf[i]

        return out_xy, out_conf
