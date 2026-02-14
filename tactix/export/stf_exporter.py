"""
Project: Tactix
File Created: 2026-02-14
Author: Xingnan Zhu
File Name: stf_exporter.py
Description:
    FIFA EPTS Standard Transfer Format (STF) exporter.
    Generates two files following the FIFA specification:
      1) match_metadata.xml — match info, team rosters, player IDs & jersey numbers
      2) match_tracking.dat — per-frame raw tracking data (positions & speeds)

    Coordinate convention:
      Tactix internal: origin top-left, units in meters (0–105 × 0–68)
      FIFA STF output: origin pitch center, units in centimeters (-5250..+5250 × -3400..+3400)

    Reference: FIFA EPTS Standard Data Format Documentation v1
    https://inside.fifa.com/technical/football-technology/standards/epts/research-development-epts-standard-data-format
"""

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from tactix.export.base import BaseExporter
from tactix.core.types import FrameData, TeamID, Point

if TYPE_CHECKING:
    from tactix.config import Config
    from tactix.core.registry import PlayerRegistry


# ==========================================================================
# Coordinate conversion: Tactix meters (top-left origin) → FIFA cm (center)
# ==========================================================================

def _to_fifa_coords(point: Point) -> Tuple[int, int]:
    """
    Convert Tactix pitch coordinates to FIFA STF coordinates.

    Tactix:  origin = top-left corner, x ∈ [0, 105] m, y ∈ [0, 68] m
    FIFA:    origin = pitch center,    x ∈ [-5250, 5250] cm, y ∈ [-3400, 3400] cm
    """
    x_cm = round((point.x - 52.5) * 100)
    y_cm = round((point.y - 34.0) * 100)
    return x_cm, y_cm


def _speed_to_cms(speed_ms: float) -> int:
    """Convert speed from m/s to cm/s."""
    return round(speed_ms * 100)


# ==========================================================================
# Resolve which "side" a goalkeeper belongs to (Home / Away)
# ==========================================================================

def _resolve_gk_side(pitch_x_m: float) -> str:
    """
    Heuristic: GK in the left half → Home, right half → Away.
    This is a simple positional fallback; the engine's registry may already
    have a more accurate assignment via voting.
    """
    return "Home" if pitch_x_m < 52.5 else "Away"


# ==========================================================================
# STF Exporter
# ==========================================================================

class StfExporter(BaseExporter):
    """
    Exports tracking data in FIFA EPTS Standard Transfer Format.

    Output:
        {output_dir}/match_metadata.xml
        {output_dir}/match_tracking.txt

    Raw data line format (per frame):
        FrameID:HomeP1,X,Y,Z,Speed;HomeP2,...;:AwayP1,X,Y,Z,Speed;AwayP2,...;:BallX,BallY,BallZ:
    """

    def __init__(self, output_dir: str, cfg: "Config", fps: int = 25) -> None:
        self.output_dir = output_dir
        self.cfg = cfg
        self.fps = fps
        self._lines: List[str] = []

        # Track all player IDs seen with their last known team, for metadata
        self._player_teams: Dict[int, TeamID] = {}

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseExporter interface
    # ------------------------------------------------------------------

    def add_frame(self, frame_data: FrameData) -> None:
        """
        Buffer one frame of tracking data as a raw STF line.

        Line format:
            FrameID:HomeP1_ID,X,Y,Z,Speed;...;:AwayP1_ID,X,Y,Z,Speed;...;:BallX,BallY,BallZ:

        Players without a valid pitch_position are omitted.
        Referees are excluded (FIFA STF tracks only team players).
        """
        frame_id = frame_data.frame_index

        home_parts: List[str] = []
        away_parts: List[str] = []

        for p in frame_data.players:
            if not p.pitch_position:
                continue
            if p.team == TeamID.REFEREE:
                continue

            # Update last-known team mapping
            self._player_teams[p.id] = p.team

            x, y = _to_fifa_coords(p.pitch_position)
            z = 0  # No height data from 2D CV
            speed = _speed_to_cms(p.speed) if p.speed else 0

            entry = f"{p.id},{x},{y},{z},{speed}"

            # Assign to home or away section
            if p.team == TeamID.A:
                home_parts.append(entry)
            elif p.team == TeamID.B:
                away_parts.append(entry)
            elif p.team == TeamID.GOALKEEPER:
                # Resolve GK to a side based on pitch position
                side = _resolve_gk_side(p.pitch_position.x)
                if side == "Home":
                    home_parts.append(entry)
                else:
                    away_parts.append(entry)
            # UNKNOWN players are skipped for STF compliance

        # Ball section
        ball_part = ""
        if frame_data.ball and frame_data.ball.pitch_position:
            bx, by = _to_fifa_coords(frame_data.ball.pitch_position)
            ball_part = f"{bx},{by},0"

        # Assemble line: FrameID:HomePlayers:AwayPlayers:Ball:
        home_str = ";".join(home_parts)
        away_str = ";".join(away_parts)
        line = f"{frame_id}:{home_str}:{away_str}:{ball_part}:"
        self._lines.append(line)

    def save(self, player_registry: Optional["PlayerRegistry"] = None) -> None:
        """
        Write metadata XML and raw tracking DAT files.

        Args:
            player_registry: If provided, used to extract confirmed jersey numbers
                             and finalized team assignments for the metadata XML.
        """
        self._write_dat()
        self._write_metadata_xml(player_registry)
        print(f"✅ FIFA STF exported to {self.output_dir}/")

    # ------------------------------------------------------------------
    # DAT writer
    # ------------------------------------------------------------------

    def _write_dat(self) -> None:
        """Write raw tracking data lines to .txt file (FIFA EPTS convention)."""
        txt_path = os.path.join(self.output_dir, "match_tracking.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in self._lines:
                f.write(line + "\n")
        print(f"   📊 Tracking data: {txt_path} ({len(self._lines)} frames)")

    # ------------------------------------------------------------------
    # Metadata XML writer
    # ------------------------------------------------------------------

    def _write_metadata_xml(self, registry: Optional["PlayerRegistry"] = None) -> None:
        """
        Generate FIFA EPTS metadata XML.

        Structure:
            <MatchSession>
              <MatchParameters>
              <GlobalConfig>
              <Players>
                <HomeTeam> ... </HomeTeam>
                <AwayTeam> ... </AwayTeam>
              </Players>
            </MatchSession>
        """
        root = ET.Element("MatchSession")

        # --- MatchParameters ---
        match_params = ET.SubElement(root, "MatchParameters")

        match_id_el = ET.SubElement(match_params, "MatchId")
        match_id_el.text = self.cfg.STF_MATCH_ID

        match_date = self.cfg.STF_MATCH_DATE or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        date_el = ET.SubElement(match_params, "MatchDate")
        date_el.text = match_date

        venue_el = ET.SubElement(match_params, "Venue")
        venue_el.text = self.cfg.STF_VENUE

        # Pitch dimensions in centimeters (FIFA convention)
        field_size = ET.SubElement(match_params, "FieldSize")
        length_el = ET.SubElement(field_size, "Length")
        length_el.text = "10500"
        width_el = ET.SubElement(field_size, "Width")
        width_el.text = "6800"

        # --- GlobalConfig ---
        global_cfg = ET.SubElement(root, "GlobalConfig")

        track_system = ET.SubElement(global_cfg, "TrackingSystem")
        track_system.text = "Tactix CV"

        track_size = ET.SubElement(global_cfg, "TrackSize")
        track_size.text = "2"  # 2D tracking

        frame_rate_el = ET.SubElement(global_cfg, "FrameRate")
        frame_rate_el.text = str(self.fps)

        total_frames_el = ET.SubElement(global_cfg, "TotalFrames")
        total_frames_el.text = str(len(self._lines))

        coord_unit = ET.SubElement(global_cfg, "CoordinateUnit")
        coord_unit.text = "cm"

        origin_el = ET.SubElement(global_cfg, "Origin")
        origin_el.text = "PitchCenter"

        # --- Players ---
        players_el = ET.SubElement(root, "Players")
        home_team_el = ET.SubElement(players_el, "HomeTeam")
        home_team_name_el = ET.SubElement(home_team_el, "TeamName")
        home_team_name_el.text = self.cfg.STF_HOME_TEAM_NAME

        away_team_el = ET.SubElement(players_el, "AwayTeam")
        away_team_name_el = ET.SubElement(away_team_el, "TeamName")
        away_team_name_el.text = self.cfg.STF_AWAY_TEAM_NAME

        # Collect player roster from registry + last-seen teams
        home_players, away_players = self._build_player_roster(registry)

        for pid, jersey in home_players:
            p_el = ET.SubElement(home_team_el, "Player")
            pid_el = ET.SubElement(p_el, "PlayerId")
            pid_el.text = str(pid)
            jersey_el = ET.SubElement(p_el, "JerseyNo")
            jersey_el.text = jersey if jersey else "?"
            team_el = ET.SubElement(p_el, "Team")
            team_el.text = "Home"

        for pid, jersey in away_players:
            p_el = ET.SubElement(away_team_el, "Player")
            pid_el = ET.SubElement(p_el, "PlayerId")
            pid_el.text = str(pid)
            jersey_el = ET.SubElement(p_el, "JerseyNo")
            jersey_el.text = jersey if jersey else "?"
            team_el = ET.SubElement(p_el, "Team")
            team_el.text = "Away"

        # --- Write XML ---
        xml_path = os.path.join(self.output_dir, "match_metadata.xml")
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        with open(xml_path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

        total_players = len(home_players) + len(away_players)
        print(f"   📋 Metadata: {xml_path} ({total_players} players)")

    # ------------------------------------------------------------------
    # Helper: build player roster from registry
    # ------------------------------------------------------------------

    def _build_player_roster(
        self, registry: Optional["PlayerRegistry"]
    ) -> Tuple[List[Tuple[int, Optional[str]]], List[Tuple[int, Optional[str]]]]:
        """
        Build home/away player lists: [(tracker_id, jersey_number), ...].

        Uses PlayerRegistry for confirmed jersey numbers and team assignments
        when available; falls back to self._player_teams for team mapping.
        """
        home: List[Tuple[int, Optional[str]]] = []
        away: List[Tuple[int, Optional[str]]] = []

        # All player IDs we've ever seen in tracking data
        all_ids = set(self._player_teams.keys())

        for pid in sorted(all_ids):
            # Try registry first for team + jersey
            jersey: Optional[str] = None
            team = self._player_teams.get(pid, TeamID.UNKNOWN)

            if registry:
                record = registry.get(pid)
                if record:
                    if record.confirmed:
                        team = record.team
                    jersey = registry.get_jersey_number(pid)

            if team == TeamID.A:
                home.append((pid, jersey))
            elif team == TeamID.B:
                away.append((pid, jersey))
            elif team == TeamID.GOALKEEPER:
                # GK without clear team — skip from roster (they'll already
                # appear via the add_frame heuristic, but for metadata we need
                # a definitive team). If registry didn't resolve it, omit.
                pass
            # UNKNOWN / REFEREE — skip

        return home, away
