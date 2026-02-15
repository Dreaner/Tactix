"""
Project: Tactix
File Created: 2026-02-14
Author: Xingnan Zhu
File Name: stf_exporter.py
Description:
    FIFA EPTS / Tracab-compatible tracking data exporter.
    Generates two files that can be loaded directly by kloppy:
      1) match_metadata.xml — Tracab flat XML (game info, periods, teams, players)
      2) match_tracking.dat — Tracab DAT format (per-frame positions & speeds)

    Usage with kloppy:
        from kloppy import tracab
        dataset = tracab.load(meta_data="match_metadata.xml", raw_data="match_tracking.dat")
        df = dataset.to_df()

    Coordinate convention:
      Tactix internal: origin top-left, units in meters (0–105 × 0–68)
      Tracab output:   origin pitch center, units in centimeters (-5250..+5250 × -3400..+3400)
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
# Coordinate conversion: Tactix meters (top-left origin) → Tracab cm (center)
# ==========================================================================

def _to_tracab_coords(point: Point) -> Tuple[int, int]:
    """
    Convert Tactix pitch coordinates to Tracab coordinates.

    Tactix:   origin = top-left corner, x ∈ [0, 105] m, y ∈ [0, 68] m
    Tracab:   origin = pitch center,    x ∈ [-5250, 5250] cm, y ∈ [-3400, 3400] cm
    """
    x_cm = round((point.x - 52.5) * 100)
    y_cm = round((point.y - 34.0) * 100)
    return x_cm, y_cm


def _speed_to_cms(speed_ms: float) -> float:
    """Convert speed from m/s to m/s (Tracab uses m/s as float)."""
    return round(speed_ms, 2)


# ==========================================================================
# Resolve which "side" a goalkeeper belongs to (Home / Away)
# ==========================================================================

def _resolve_gk_team_id(pitch_x_m: float) -> int:
    """
    Heuristic: GK in the left half → Home (1), right half → Away (0).
    Tracab uses 1=Home, 0=Away.
    """
    return 1 if pitch_x_m < 52.5 else 0


# ==========================================================================
# STF Exporter (Tracab-compatible)
# ==========================================================================

class StfExporter(BaseExporter):
    """
    Exports tracking data in Tracab DAT format, compatible with kloppy.

    Output:
        {output_dir}/match_metadata.xml  — Tracab flat XML metadata
        {output_dir}/match_tracking.dat  — Tracab DAT raw tracking data

    DAT line format (per frame):
        FrameID:TeamID,PlayerID,JerseyNo,X,Y,Speed;...;:BallX,BallY,BallZ,BallSpeed,BallOwner,BallState;:
    """

    def __init__(self, output_dir: str, cfg: "Config", fps: int = 25) -> None:
        self.output_dir = output_dir
        self.cfg = cfg
        self.fps = fps
        self._lines: List[str] = []

        # Track all player IDs seen, with last known team and jersey number
        self._player_teams: Dict[int, TeamID] = {}
        self._player_jerseys: Dict[int, str] = {}  # tracker_id → jersey_no string

        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseExporter interface
    # ------------------------------------------------------------------

    def add_frame(self, frame_data: FrameData) -> None:
        """
        Buffer one frame as a Tracab DAT line.

        Tracab DAT line:
            FrameID:TeamID,PlayerID,JerseyNo,X,Y,Speed;...;:BallX,BallY,BallZ,BallSpeed,OwnerTeam,BallState;:

        TeamID: 1=Home(TeamA), 0=Away(TeamB), -1=undefined
        BallOwner: "H"=Home, "A"=Away
        BallState: "Alive" or "Dead"

        Players without valid pitch_position are omitted.
        Referees are excluded (Tracab uses team_id 3/4 for officials, kloppy skips them).
        """
        frame_id = frame_data.frame_index
        player_parts: List[str] = []

        for p in frame_data.players:
            if not p.pitch_position:
                continue
            if p.team == TeamID.REFEREE:
                continue

            # Determine Tracab team_id: 1=Home(A), 0=Away(B)
            if p.team == TeamID.A:
                team_id = 1
            elif p.team == TeamID.B:
                team_id = 0
            elif p.team == TeamID.GOALKEEPER:
                team_id = _resolve_gk_team_id(p.pitch_position.x)
            else:
                continue  # Skip UNKNOWN

            # Update last-known mappings
            self._player_teams[p.id] = p.team

            # Determine jersey number: use player's jersey_number if available, else tracker_id
            jersey_no = p.jersey_number if p.jersey_number else str(p.id)
            self._player_jerseys[p.id] = jersey_no

            x, y = _to_tracab_coords(p.pitch_position)
            speed = _speed_to_cms(p.speed) if p.speed else 0.0

            # Tracab format: TeamID,PlayerID,JerseyNo,X,Y,Speed
            entry = f"{team_id},{p.id},{jersey_no},{x},{y},{speed}"
            player_parts.append(entry)

        # Ball section: X,Y,Z,Speed,OwnerTeam,BallState
        ball = frame_data.ball
        if ball and ball.pitch_position:
            bx, by = _to_tracab_coords(ball.pitch_position)
            ball_speed = 0.0  # Ball speed not tracked yet
            # Determine ball owner team for Tracab
            owner_team = "H"  # Default to Home
            if ball.owner_id is not None:
                owner_player_team = self._player_teams.get(ball.owner_id, TeamID.A)
                owner_team = "A" if owner_player_team == TeamID.B else "H"
            ball_part = f"{bx},{by},0,{ball_speed},{owner_team},Alive"
        else:
            # No ball detected — place at center with Dead state
            ball_part = "0,0,0,0.0,H,Dead"

        # Assemble Tracab DAT line:
        # FrameID:Player1;Player2;...;:BallX,BallY,BallZ,Speed,Owner,State;:
        players_str = ";".join(player_parts) + ";" if player_parts else ""
        line = f"{frame_id}:{players_str}:{ball_part};:"
        self._lines.append(line)

    def save(self, player_registry: Optional["PlayerRegistry"] = None) -> None:
        """
        Write metadata XML and raw tracking DAT files.

        Args:
            player_registry: If provided, used to extract confirmed jersey numbers
                             and finalized team assignments for the metadata XML.
        """
        # Resolve final jersey numbers from registry before writing
        if player_registry:
            self._resolve_jerseys_from_registry(player_registry)

        self._write_dat()
        self._write_metadata_xml(player_registry)
        print(f"✅ FIFA STF (Tracab-compatible) exported to {self.output_dir}/")

    # ------------------------------------------------------------------
    # DAT writer
    # ------------------------------------------------------------------

    def _write_dat(self) -> None:
        """Write raw tracking data lines to .dat file (Tracab convention)."""
        dat_path = os.path.join(self.output_dir, "match_tracking.dat")
        with open(dat_path, "w", encoding="utf-8") as f:
            for line in self._lines:
                f.write(line + "\n")
        print(f"   📊 Tracking data: {dat_path} ({len(self._lines)} frames)")

    # ------------------------------------------------------------------
    # Metadata XML writer (Tracab flat XML format)
    # ------------------------------------------------------------------

    def _write_metadata_xml(self, registry: Optional["PlayerRegistry"] = None) -> None:
        """
        Generate Tracab-compatible flat XML metadata.

        kloppy expects this structure:
            <root>
                <GameID>...</GameID>
                <FrameRate>25</FrameRate>
                <PitchLongSide>10500</PitchLongSide>
                <PitchShortSide>6800</PitchShortSide>
                <Phase1StartFrame>0</Phase1StartFrame>
                <Phase1EndFrame>N</Phase1EndFrame>
                <HomeTeam>
                    <TeamId>home</TeamId>
                    <ShortName>Home</ShortName>
                    <Players>
                        <item>
                            <PlayerId>1</PlayerId>
                            <JerseyNo>7</JerseyNo>
                            <FirstName>Player</FirstName>
                            <LastName>7</LastName>
                            <StartFrameCount>0</StartFrameCount>
                            <StartingPosition>M</StartingPosition>
                        </item>
                    </Players>
                </HomeTeam>
                <AwayTeam> ... </AwayTeam>
            </root>
        """
        root = ET.Element("TracabMetaData")

        # --- Game info ---
        _text_el(root, "GameID", self.cfg.STF_MATCH_ID)

        kickoff = self.cfg.STF_MATCH_DATE or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _text_el(root, "Kickoff", kickoff)

        _text_el(root, "FrameRate", str(self.fps))

        # Pitch dimensions in centimeters
        _text_el(root, "PitchLongSide", "10500")
        _text_el(root, "PitchShortSide", "6800")

        # --- Periods (phases) ---
        # Treat the entire video as a single period (Phase 1)
        total_frames = len(self._lines)
        start_frame = 0
        end_frame = total_frames - 1 if total_frames > 0 else 0

        _text_el(root, "Phase1StartFrame", str(start_frame))
        _text_el(root, "Phase1EndFrame", str(end_frame))
        # Phases 2-5 set to 0 (no second half / extra time distinction)
        for i in range(2, 6):
            _text_el(root, f"Phase{i}StartFrame", "0")
            _text_el(root, f"Phase{i}EndFrame", "0")

        # --- Teams & Players ---
        home_players, away_players = self._build_player_roster(registry)

        home_team_el = ET.SubElement(root, "HomeTeam")
        _text_el(home_team_el, "TeamID", "home")
        _text_el(home_team_el, "ShortName", self.cfg.STF_HOME_TEAM_NAME)
        home_players_el = ET.SubElement(home_team_el, "Players")
        for pid, jersey_no in home_players:
            _add_player_item(home_players_el, pid, jersey_no, start_frame)

        away_team_el = ET.SubElement(root, "AwayTeam")
        _text_el(away_team_el, "TeamID", "away")
        _text_el(away_team_el, "ShortName", self.cfg.STF_AWAY_TEAM_NAME)
        away_players_el = ET.SubElement(away_team_el, "Players")
        for pid, jersey_no in away_players:
            _add_player_item(away_players_el, pid, jersey_no, start_frame)

        # --- Write XML ---
        xml_path = os.path.join(self.output_dir, "match_metadata.xml")
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        with open(xml_path, "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)

        total_players = len(home_players) + len(away_players)
        print(f"   📋 Metadata: {xml_path} ({total_players} players)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_jerseys_from_registry(self, registry: "PlayerRegistry") -> None:
        """Pull confirmed jersey numbers from registry into _player_jerseys."""
        for pid in list(self._player_teams.keys()):
            confirmed_jersey = registry.get_jersey_number(pid)
            if confirmed_jersey:
                self._player_jerseys[pid] = confirmed_jersey

    def _build_player_roster(
        self, registry: Optional["PlayerRegistry"]
    ) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
        """
        Build home/away player lists: [(tracker_id, jersey_number_str), ...].

        Uses PlayerRegistry for confirmed jersey numbers and team assignments
        when available; falls back to self._player_teams for team mapping.
        """
        home: List[Tuple[int, str]] = []
        away: List[Tuple[int, str]] = []

        all_ids = set(self._player_teams.keys())

        for pid in sorted(all_ids):
            jersey = self._player_jerseys.get(pid, str(pid))
            team = self._player_teams.get(pid, TeamID.UNKNOWN)

            if registry:
                record = registry.get(pid)
                if record and record.confirmed:
                    team = record.team
                confirmed_jersey = registry.get_jersey_number(pid)
                if confirmed_jersey:
                    jersey = confirmed_jersey

            if team == TeamID.A:
                home.append((pid, jersey))
            elif team == TeamID.B:
                away.append((pid, jersey))
            elif team == TeamID.GOALKEEPER:
                # GK — omit from roster if team not resolved
                pass

        return home, away


# ==========================================================================
# XML helper functions
# ==========================================================================

def _text_el(parent: ET.Element, tag: str, text: str) -> ET.Element:
    """Create a child element with text content."""
    el = ET.SubElement(parent, tag)
    el.text = text
    return el


def _add_player_item(
    players_el: ET.Element, player_id: int, jersey_no: str, start_frame: int
) -> None:
    """
    Add a <item> player element matching kloppy's Tracab flat XML expectation.

    <item>
        <PlayerID>1</PlayerID>
        <JerseyNo>7</JerseyNo>
        <FirstName>Player</FirstName>
        <LastName>7</LastName>
        <StartFrameCount>0</StartFrameCount>
        <StartingPosition>M</StartingPosition>
    </item>
    """
    item = ET.SubElement(players_el, "item")
    _text_el(item, "PlayerID", str(player_id))
    _text_el(item, "JerseyNo", str(jersey_no))
    _text_el(item, "FirstName", "Player")
    # LastName must NOT be a pure number — lxml.objectify converts numeric-only
    # text to IntElement, breaking kloppy's string concatenation.
    _text_el(item, "LastName", f"#{jersey_no}")
    _text_el(item, "StartFrameCount", str(start_frame))
    _text_el(item, "StartingPosition", "M")  # Default to Midfielder
