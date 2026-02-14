"""
Project: Tactix
File Created: 2026-02-14
Author: Xingnan Zhu
File Name: pdf_exporter.py
Description:
    Generates a multi-page post-match tactical PDF report from accumulated
    analytics data.  Subclasses BaseExporter — accumulates cheap per-frame
    counters in add_frame(), then pulls analytics module summaries and
    renders overlay images at save() time.

    Requires: reportlab, Pillow (added to requirements.txt).
"""

import io
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from tactix.config import Config
from tactix.core.types import FrameData, TeamID
from tactix.export.base import BaseExporter

try:
    from reportlab.lib import colors as rl_colors  # type: ignore[import-untyped]
    from reportlab.lib.enums import TA_CENTER, TA_LEFT  # type: ignore[import-untyped]
    from reportlab.lib.pagesizes import A4  # type: ignore[import-untyped]
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import-untyped]
    from reportlab.lib.units import cm, mm  # type: ignore[import-untyped]
    from reportlab.lib.utils import ImageReader  # type: ignore[import-untyped]
    from reportlab.platypus import (  # type: ignore[import-untyped]
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


class PdfReportExporter(BaseExporter):
    """
    Generates a post-match PDF report from accumulated analytics data.

    Workflow:
        1. Engine calls ``attach_analytics(...)`` once after init.
        2. Engine calls ``add_frame(frame_data)`` every frame (cheap counters).
        3. Engine calls ``save()`` once at the end to render the PDF.
    """

    def __init__(self, output_path: str, cfg: Config) -> None:
        self.output_path = output_path
        self.cfg = cfg

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # --- per-frame counters ---
        self._frame_count: int = 0
        self._possession_frames: Dict[TeamID, int] = defaultdict(int)
        self._total_passes: Dict[TeamID, int] = defaultdict(int)
        self._total_shots: Dict[TeamID, int] = defaultdict(int)
        self._total_duels: int = 0
        self._total_corners: Dict[TeamID, int] = defaultdict(int)
        self._total_free_kicks: int = 0
        self._last_owner_team: Optional[TeamID] = None

        # --- analytics references (set via attach_analytics) ---
        self._shot_map: Any = None
        self._zone_analyzer: Any = None
        self._pass_sonar: Any = None
        self._buildup_tracker: Any = None
        self._transition_tracker: Any = None
        self._duel_heatmap: Any = None
        self._corner_analyzer: Any = None
        self._free_kick_analyzer: Any = None
        self._heatmap: Any = None
        self._formation_detector: Any = None

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    def attach_analytics(
        self,
        *,
        shot_map: Any = None,
        zone_analyzer: Any = None,
        pass_sonar: Any = None,
        buildup_tracker: Any = None,
        transition_tracker: Any = None,
        duel_heatmap: Any = None,
        corner_analyzer: Any = None,
        free_kick_analyzer: Any = None,
        heatmap: Any = None,
        formation_detector: Any = None,
    ) -> None:
        """Bind references to analytics modules for use in save()."""
        self._shot_map = shot_map
        self._zone_analyzer = zone_analyzer
        self._pass_sonar = pass_sonar
        self._buildup_tracker = buildup_tracker
        self._transition_tracker = transition_tracker
        self._duel_heatmap = duel_heatmap
        self._corner_analyzer = corner_analyzer
        self._free_kick_analyzer = free_kick_analyzer
        self._heatmap = heatmap
        self._formation_detector = formation_detector

    # ------------------------------------------------------------------
    # Per-frame accumulation (BaseExporter interface)
    # ------------------------------------------------------------------

    def add_frame(self, frame_data: FrameData) -> None:
        """Accumulate lightweight per-frame summary counters."""
        self._frame_count += 1

        # Possession tracking by ball owner
        if frame_data.ball and frame_data.ball.owner_id is not None:
            owner = frame_data.get_player_by_id(frame_data.ball.owner_id)
            if owner and owner.team in (TeamID.A, TeamID.B):
                self._possession_frames[owner.team] += 1
                self._last_owner_team = owner.team
            elif self._last_owner_team is not None:
                self._possession_frames[self._last_owner_team] += 1

        events = frame_data.events
        if events is None:
            return
        for p in events.passes:
            if p.team in (TeamID.A, TeamID.B):
                self._total_passes[p.team] += 1
        for s in events.shots:
            if s.team in (TeamID.A, TeamID.B):
                self._total_shots[s.team] += 1
        self._total_duels += len(events.duels)
        if events.corner:
            self._total_corners[events.corner.attacking_team] += 1
        if events.free_kick:
            self._total_free_kicks += 1

    # ------------------------------------------------------------------
    # PDF Generation
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Render the multi-page PDF report."""
        if not _HAS_REPORTLAB:
            print("❌ reportlab is not installed. Run: pip install reportlab Pillow")
            return

        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TactixTitle", parent=styles["Title"],
            fontSize=22, spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "TactixH2", parent=styles["Heading2"],
            fontSize=14, spaceAfter=8, spaceBefore=14,
        )
        body_style = styles["BodyText"]

        story: list = []

        # ── PAGE 1: Match Summary ─────────────────────────────
        story.append(Paragraph("Tactix Match Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; "
            f"Video: {os.path.basename(self.cfg.INPUT_VIDEO)} &nbsp;|&nbsp; "
            f"Frames: {self._frame_count}",
            body_style,
        ))
        story.append(Spacer(1, 8 * mm))

        # Possession
        story.append(Paragraph("Possession", heading_style))
        total_poss = max(sum(self._possession_frames.values()), 1)
        poss_a = self._possession_frames.get(TeamID.A, 0) / total_poss * 100
        poss_b = self._possession_frames.get(TeamID.B, 0) / total_poss * 100
        story.append(Paragraph(
            f"Team A: <b>{poss_a:.1f}%</b> &nbsp;&nbsp; Team B: <b>{poss_b:.1f}%</b>",
            body_style,
        ))
        story.append(Spacer(1, 4 * mm))

        # Formation
        if self._formation_detector is not None:
            story.append(Paragraph("Detected Formation", heading_style))
            fm = self._formation_detector.summary()
            story.append(Paragraph(
                f"Team A: <b>{fm['team_a']}</b> (confidence {fm['team_a_confidence']:.0%}) "
                f"&nbsp;&nbsp; "
                f"Team B: <b>{fm['team_b']}</b> (confidence {fm['team_b_confidence']:.0%})",
                body_style,
            ))
            story.append(Spacer(1, 4 * mm))

        # Key Stats Table
        story.append(Paragraph("Key Statistics", heading_style))
        stats_data = [
            ["Metric", "Team A", "Team B"],
            ["Passes", str(self._total_passes.get(TeamID.A, 0)),
             str(self._total_passes.get(TeamID.B, 0))],
            ["Shots", str(self._total_shots.get(TeamID.A, 0)),
             str(self._total_shots.get(TeamID.B, 0))],
            ["Corners", str(self._total_corners.get(TeamID.A, 0)),
             str(self._total_corners.get(TeamID.B, 0))],
            ["Total Duels", str(self._total_duels), ""],
            ["Free Kicks", str(self._total_free_kicks), ""],
        ]
        story.append(self._make_table(stats_data, Table, TableStyle, rl_colors))

        story.append(PageBreak())

        # ── PAGE 2: Attacking Analysis ────────────────────────
        story.append(Paragraph("Attacking Analysis", title_style))

        # Shot Map
        if self._shot_map is not None:
            story.append(Paragraph("Shot Map", heading_style))
            sm = self._shot_map.summary()
            story.append(Paragraph(
                f"Total shots: {sm.get('total', 0)} &nbsp;|&nbsp; "
                f"On target: {sm.get('on_target', 0)}",
                body_style,
            ))
            img = self._render_overlay_image("shot_map")
            if img is not None:
                story.append(Spacer(1, 4 * mm))
                story.append(img)
            story.append(Spacer(1, 6 * mm))

        # Zone 14
        if self._zone_analyzer is not None:
            story.append(Paragraph("Zone 14 Analysis", heading_style))
            z14 = self._zone_analyzer.summary()
            story.append(Paragraph(
                f"Zone 14 passes — Team A: {z14.get('team_a', 0)}, Team B: {z14.get('team_b', 0)}",
                body_style,
            ))
            img = self._render_overlay_image("zone_14")
            if img is not None:
                story.append(Spacer(1, 4 * mm))
                story.append(img)
            story.append(Spacer(1, 6 * mm))

        # Build-up
        if self._buildup_tracker is not None:
            story.append(Paragraph("Build-up Play", heading_style))
            bu = self._buildup_tracker.summary()
            bu_data = [
                ["Metric", "Value"],
                ["Total sequences", str(bu.get("total", len(self._buildup_tracker.sequences)))],
                ["Avg passes / sequence",
                 f"{bu.get('avg_passes', 0):.1f}" if "avg_passes" in bu else self._avg_buildup_passes()],
            ]
            story.append(self._make_table(bu_data, Table, TableStyle, rl_colors))

        story.append(PageBreak())

        # ── PAGE 3: Transition & Defense ──────────────────────
        story.append(Paragraph("Transition & Defense", title_style))

        # Transition
        if self._transition_tracker is not None:
            story.append(Paragraph("Transitions", heading_style))
            tr = self._transition_tracker.summary()
            tr_data = [
                ["Metric", "Team A", "Team B"],
                ["Counter-attacks",
                 str(tr.get("team_a_attacks", 0)), str(tr.get("team_b_attacks", 0))],
                ["Defensive recoveries",
                 str(tr.get("team_a_defenses", 0)), str(tr.get("team_b_defenses", 0))],
            ]
            story.append(self._make_table(tr_data, Table, TableStyle, rl_colors))
            story.append(Spacer(1, 6 * mm))

        # Duel Heatmap
        if self._duel_heatmap is not None:
            story.append(Paragraph("Duel Heatmap", heading_style))
            dh = self._duel_heatmap.summary()
            story.append(Paragraph(f"Total duels recorded: {dh.get('total', 0)}", body_style))
            img = self._render_overlay_image("duel_heatmap")
            if img is not None:
                story.append(Spacer(1, 4 * mm))
                story.append(img)

        story.append(PageBreak())

        # ── PAGE 4: Set Pieces & Heatmaps ─────────────────────
        story.append(Paragraph("Set Pieces & Heatmaps", title_style))

        # Corners
        if self._corner_analyzer is not None:
            story.append(Paragraph("Corner Kicks", heading_style))
            corner_data = [
                ["Metric", "Team A", "Team B"],
                ["Corners taken",
                 str(self._total_corners.get(TeamID.A, 0)),
                 str(self._total_corners.get(TeamID.B, 0))],
            ]
            story.append(self._make_table(corner_data, Table, TableStyle, rl_colors))
            story.append(Spacer(1, 4 * mm))

        # Free kicks
        if self._free_kick_analyzer is not None:
            story.append(Paragraph("Free Kicks", heading_style))
            story.append(Paragraph(f"Total free kicks: {self._total_free_kicks}", body_style))
            story.append(Spacer(1, 6 * mm))

        # Heatmap image
        if self._heatmap is not None:
            story.append(Paragraph("Team Heatmaps", heading_style))
            img = self._render_overlay_image("heatmap")
            if img is not None:
                story.append(img)

        # ── Build PDF ─────────────────────────────────────────
        try:
            doc.build(story)
            print(f"📄 PDF report saved to {self.output_path}")
        except Exception as e:
            print(f"❌ Failed to generate PDF report: {e}")

    # ------------------------------------------------------------------
    # Overlay image rendering helpers
    # ------------------------------------------------------------------

    def _render_overlay_image(self, kind: str) -> Any:
        """
        Render an analytics overlay into a reportlab Image flowable.
        Returns None if the overlay cannot be generated.
        """
        try:
            arr = self._get_overlay_array(kind)
            if arr is None:
                return None
            return self._ndarray_to_rl_image(arr, width=14 * cm)
        except Exception:
            return None

    def _get_overlay_array(self, kind: str) -> Optional[np.ndarray]:
        """Render overlay to numpy array by calling the corresponding overlay class."""
        from tactix.core.types import PitchConfig
        cw = PitchConfig.PIXEL_WIDTH
        ch = PitchConfig.PIXEL_HEIGHT

        try:
            if kind == "shot_map" and self._shot_map is not None:
                from tactix.visualization.overlays.attacking.shot_map import ShotMapOverlay
                return ShotMapOverlay.render(self._shot_map, cw, ch)
            if kind == "zone_14" and self._zone_analyzer is not None:
                from tactix.visualization.overlays.attacking.zone_14 import Zone14Overlay
                return Zone14Overlay.render(self._zone_analyzer, cw, ch)
            if kind == "duel_heatmap" and self._duel_heatmap is not None:
                from tactix.visualization.overlays.defense.duel_heatmap import DuelHeatmapOverlay
                return DuelHeatmapOverlay.render(self._duel_heatmap)
            if kind == "heatmap" and self._heatmap is not None:
                from tactix.visualization.overlays.base.heatmap import HeatmapOverlay
                return HeatmapOverlay.render(self._heatmap)
        except Exception:
            return None
        return None

    @staticmethod
    def _ndarray_to_rl_image(arr: np.ndarray, width: float) -> Any:
        """Convert an RGBA/BGR numpy array to a reportlab Image flowable."""
        # Convert to RGB
        if arr.ndim == 3 and arr.shape[2] == 4:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            rgb = arr

        # Encode to PNG in memory
        success, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not success:
            return None
        bio = io.BytesIO(buf.tobytes())
        bio.seek(0)

        h, w = arr.shape[:2]
        aspect = h / max(w, 1)
        return RLImage(bio, width=width, height=width * aspect)

    # ------------------------------------------------------------------
    # Table helper
    # ------------------------------------------------------------------

    @staticmethod
    def _make_table(data: List[List[str]], Table: type,
                    TableStyle: type, rl_colors: Any) -> Any:
        """Build a styled reportlab Table from 2D string data."""
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1D3557")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("BACKGROUND", (0, 1), (-1, -1), rl_colors.HexColor("#F1FAEE")),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("TOPPADDING", (0, 1), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
        ]))
        return table

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def _avg_buildup_passes(self) -> str:
        """Compute average passes per build-up sequence."""
        if self._buildup_tracker is None or not self._buildup_tracker.sequences:
            return "0"
        total = sum(s.pass_count for s in self._buildup_tracker.sequences)
        return f"{total / len(self._buildup_tracker.sequences):.1f}"
