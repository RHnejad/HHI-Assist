"""
hhi_utils.py - Shared utilities for HHI-G MJCF generation and visualization.

Contains:
- BVHParser: Robust BVH file parser (class-based, similar to HHI-Assist/utils/FK.py)
- Configuration loading (load_config)
- Bone mapping dictionaries (BONE_MAPPING, COLORS, RADIUS)
- Coordinate transformation matrix (T_ROOT)
"""

import numpy as np
import yaml
import os
from typing import List, Optional, Dict, Any

# =============================================================================
# BONE MAPPING CONFIGURATION
# =============================================================================

# BONE_MAPPING: BVH bone name -> {link: MJCF body name, joint: MJCF joint prefix}
# Must match generate_hhi_mjcf.py BONE_MAP exactly
BONE_MAPPING: Dict[str, Dict[str, Optional[str]]] = {
    "Hips":          {"link": "Pelvis/Base",        "joint": None,  "parent": None, "children": ["Spine", "LeftUpLeg", "RightUpLeg"]},          
    "Spine":         {"link": "SpineSacralL",      "joint": "SpineSacralJ", "parent": "Hips", "children": ["Spine1"]},
    "Spine1":        {"link": "SpineLumbarL",      "joint": "SpineLumbarJ", "parent": "Spine", "children": ["Neck", "LeftShoulder", "RightShoulder"]},
    
    "Neck":          {"link": "SpineThoraxL",       "joint": "SpineThoraxJ", "parent": "Spine1", "children": ["Head"]},
    "Head":          {"link": "HeadL",       "joint": "HeadJ", "parent": "Neck", "children": ["EndSite"]},
    
    "LeftShoulder":  {"link": "LClavicleL",  "joint": "LClavicleJ", "parent": "Spine1", "children": ["LeftArm"]},
    "LeftArm":       {"link": "LeftShoulderBone",  "joint": "LShoulderJ", "parent": "LeftShoulder", "children": ["LeftForeArm"]},
    "LeftForeArm":   {"link": "LeftUpperArm",   "joint": "LElbowJ", "parent": "LeftArm", "children": ["LeftHand"]},
    "LeftHand":      {"link": "LeftForeArm",      "joint": "LWristJ", "parent": "LeftForeArm", "children": ["EndSite"]},
    
    "RightShoulder": {"link": "RClavicleL",  "joint": "RClavicleJ", "parent": "Spine1", "children": ["RightArm"]},
    "RightArm":      {"link": "RightShoulderBone",  "joint": "RShoulderJ", "parent": "RightShoulder", "children": ["RightForeArm"]},
    "RightForeArm":  {"link": "RightUpperArm",   "joint": "RElbowJ", "parent": "RightArm", "children": ["RightHand"]},
    "RightHand":     {"link": "RightForeArm",      "joint": "RWristJ", "parent": "RightForeArm", "children": ["EndSite"]},
    
    "LeftUpLeg":     {"link": "LHipL",     "joint": "PelvisLJ", "parent": "Hips", "children": ["LeftLeg"]},
    "LeftLeg":       {"link": "LThighL",      "joint": "LHipJ", "parent": "LeftUpLeg", "children": ["LeftFoot"]},   
    "LeftFoot":      {"link": "LShinL",      "joint": "LKneeJ", "parent": "LeftLeg", "children": ["LeftToeBase"]},
    "LeftToeBase":   {"link": "LFootL",       "joint": "LAnkleJ", "parent": "LeftFoot", "children": ["EndSite"]},
    
    "RightUpLeg":    {"link": "RHipL",     "joint": "PelvisRJ", "parent": "Hips", "children": ["RightLeg"]},
    "RightLeg":      {"link": "RThighL",      "joint": "RHipJ", "parent": "RightUpLeg", "children": ["RightFoot"]},
    "RightFoot":     {"link": "RShinL",      "joint": "RKneeJ", "parent": "RightLeg", "children": ["RightToeBase"]},
    "RightToeBase":  {"link": "RFootL",       "joint": "RAnkleJ", "parent": "RightFoot", "children": ["EndSite"]}, 
}

INCLUDED_NODES = set(BONE_MAPPING.keys())


def get_color(role: str, bone_name: str) -> str:
    """
    Returns RGBA string based on Role (cg/cr) and Side (Left/Center/Right).
    CG: Blue theme. Left (Light), Center (Med), Right (Dark).
    CR: Pink theme. Left (Light), Center (Med), Right (Dark).
    """
    # 1. Determine Side
    if "Left" in bone_name or bone_name.startswith("L"):
        side = "Left"
    elif "Right" in bone_name or bone_name.startswith("R"):
        side = "Right"
    else:
        side = "Center"
        
    # 2. Assign Color
    if role == "cg":
        # Blue Theme
        if side == "Left":   return "0.5 0.7 1.0 1"  # Light Blue
        if side == "Right":  return "0.1 0.2 0.5 1"  # Dark Blue
        return "0.2 0.4 0.8 1"                       # Medium Blue (Center)
        
    elif role == "cr":
        # Pink Theme
        if side == "Left":   return "1.0 0.7 0.85 1" # Light Pink
        if side == "Right":  return "0.6 0.2 0.4 1"  # Dark Pink
        return "0.9 0.4 0.6 1"                       # Medium Pink (Center)
    
    return "0.5 0.5 0.5 1" # Default Grey


RADIUS: Dict[str, float] = { 
    "Hips": 0.088,
    "Spine": 0.077, "Spine1": 0.077,
    "Neck": 0.033, "Head": 0.04,
    "LeftShoulder": 0.044, "LeftArm": 0.044, "LeftForeArm": 0.044, "LeftHand": 0.033,
    "RightShoulder": 0.044, "RightArm": 0.044, "RightForeArm": 0.044, "RightHand": 0.033,
    "LeftUpLeg": 0.066, "LeftLeg": 0.055, "LeftFoot": 0.044, "LeftToeBase": 0.0275,
    "RightUpLeg": 0.066, "RightLeg": 0.055, "RightFoot": 0.044, "RightToeBase": 0.0275,
}

def get_radius(bone_name: str) -> float:
    """Returns the radius for a given bone, defaulting to 0.04."""
    return RADIUS.get(bone_name, 0.04)

# Coordinate transform: Y-up (BVH) -> Z-up (MJCF/Newton)
T_ROOT = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float64)

def get_bone_info(raw_name: str) -> Optional[Dict[str, Optional[str]]]:
    """Helper to get mapping for a bone name."""
    clean = raw_name.split(':')[-1]
    return BONE_MAPPING.get(clean)


# =============================================================================
# CONFIG LOADER
# =============================================================================

def load_config(config_path: str = "hhi_config.yaml") -> Dict[str, Any]:
    """
    Loads the YAML config file.
    
    Returns:
        dict with keys: 'cg', 'cr' (file paths), 'name' (dataset name)
    """
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = config.get("current_dataset", {})
    bvh_files = {
        "cg": dataset.get("cg_path"),
        "cr": dataset.get("cr_path"),
        "name": dataset.get("name", "default")
    }
    
    if not bvh_files["cg"] or not bvh_files["cr"]:
        raise ValueError("Config must specify 'cg_path' and 'cr_path' under 'current_dataset'.")
        
    return bvh_files


def get_bone_info(raw_name: str) -> Optional[Dict[str, Optional[str]]]:
    """Returns the mapping dict {'link':..., 'joint':...} for a raw bone name."""
    clean = raw_name.split(':')[-1]
    return BONE_MAPPING.get(clean)


# =============================================================================
# BVH PARSER CLASS
# =============================================================================

class BVHParser:
    """
    Robust BVH file parser.
    
    Usage:
        parser = BVHParser("/path/to/file.bvh")
        parser.parse()  # or parser.parse(read_motion=False) for hierarchy only
        
        # Access data:
        parser.bones        # List of bone names
        parser.parents      # List of parent indices (-1 for root)
        parser.offsets      # List of offset vectors (np.ndarray)
        parser.orders       # List of rotation orders (e.g., "zxy")
        parser.positions    # Root positions (Frames x 3)
        parser.rotations    # Joint rotations (Frames x Bones x 3) as [X, Y, Z]
        parser.frametime    # Time per frame in seconds
    """
    
    def __init__(self, filename: str):
        self.filename = filename
        
        # Hierarchy data
        self.bones: List[str] = []
        self.parents: List[int] = []
        self.orders: List[Optional[str]] = []
        self.offsets: List[np.ndarray] = []
        self.channel_map: List[tuple] = []
        
        # Motion data
        self.positions: Optional[np.ndarray] = None
        self.rotations: Optional[np.ndarray] = None
        self.frametime: float = 0.033
        self.num_frames: int = 0
    
    def parse(self, read_motion: bool = True) -> 'BVHParser':
        """
        Parse the BVH file.
        
        Args:
            read_motion: If True, also parse motion data. If False, only parse hierarchy.
        
        Returns:
            self (for method chaining)
        """
        self._parse_hierarchy()
        if read_motion:
            self._parse_motion()
        return self
    
    def _parse_hierarchy(self) -> None:
        """Parses the skeleton hierarchy from the BVH file."""
        print(f"Parsing BVH Hierarchy: {self.filename}...")
        
        with open(self.filename, 'r') as f:
            lines = []
            for line in f:
                if line.strip().startswith("MOTION"):
                    break
                lines.append(line)
        
        stack = [-1]  # Parent index stack
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('ROOT') or line.startswith('JOINT'):
                bone_name = line.split()[1]
                parent_idx = stack[-1]
                current_idx = len(self.bones)
                
                # Assertion: Verify Hierarchy against BONE_MAPPING
                clean_name = bone_name.split(':')[-1]
                mapping = BONE_MAPPING.get(clean_name)
                
                if mapping:
                    if parent_idx != -1:
                        actual_parent = self.bones[parent_idx].split(':')[-1]
                        expected_parent = mapping.get("parent")
                        if expected_parent and actual_parent != expected_parent:
                            print(f"[WARNING] Hierarchy Mismatch! Bone: {clean_name}, BVH Parent: {actual_parent}, Expected: {expected_parent}")
                            # raise ValueError(f"Hierarchy mismatch for {clean_name}") # User asked to 'assert', but maybe warning is safer to not crash on trivial diffs? 
                            # User said "assert". I will raise error if mismatch implies structural failure.
                            assert actual_parent == expected_parent, f"Hierarchy mismatch for {clean_name}: BVH has {actual_parent}, expected {expected_parent}"
                    else:
                        # Root check
                        expected_parent = mapping.get("parent")
                        assert expected_parent is None, f"Root {clean_name} should have parent {expected_parent}"

                self.bones.append(bone_name)
                self.parents.append(parent_idx)
                self.orders.append(None)
                self.offsets.append(np.zeros(3))
                stack.append(current_idx)
                
            elif line.startswith('End Site'):
                parent_idx = stack[-1]
                
                # Assertion: Verify Parent expects EndSite
                if parent_idx != -1:
                    parent_raw = self.bones[parent_idx]
                    parent_clean = parent_raw.split(':')[-1]
                    p_mapping = BONE_MAPPING.get(parent_clean)
                    if p_mapping:
                        # Check if 'EndSite' is in expected children (case insensitive or exact string?)
                        # Mapping uses "EndSite" string in children list.
                        # Wait, user mapping has "children": ["Spine", ...].
                        # And for Head: "children": ["EndSite"].
                        expected_children = p_mapping.get("children", [])
                        # "EndSite" usually not named in BVH "End Site" line (it's just "End Site").
                        # We verify parent allows it.
                        assert "EndSite" in expected_children, f"Unexpected End Site for parent {parent_clean}. Mapping children: {expected_children}"

                self.bones.append("End Site")
                self.parents.append(parent_idx)
                self.orders.append(None)
                self.offsets.append(np.zeros(3))
                stack.append(len(self.bones) - 1)
                
            elif line.startswith('OFFSET'):
                vals = [float(x) for x in line.split()[1:]]
                self.offsets[stack[-1]] = np.array(vals)
                
            elif line.startswith('CHANNELS'):
                parts = line.split()
                channels = parts[2:]
                current_idx = stack[-1]
                
                rot_order = ""
                for ch in channels:
                    self.channel_map.append((current_idx, ch))
                    if 'rotation' in ch.lower():
                        rot_order += ch[0].lower()
                
                if rot_order:
                    self.orders[current_idx] = rot_order
                    
            elif line.startswith('}'):
                stack.pop()
    
    def _parse_motion(self) -> None:
        """Parses motion data from the BVH file."""
        print(f"Parsing BVH Motion: {self.filename}...")
        
        with open(self.filename, 'r') as f:
            content = f.read()
        
        parts = content.split("MOTION")
        if len(parts) < 2:
            print("  No MOTION section found.")
            return
        
        motion_lines = parts[1].strip().split('\n')
        
        # Parse headers
        data_start_line = 0
        for i, line in enumerate(motion_lines):
            if line.startswith('Frames:'):
                self.num_frames = int(line.split()[1])
            elif line.startswith('Frame Time:'):
                self.frametime = float(line.split()[2])
            elif line and (line[0].isdigit() or line[0] == '-'):
                data_start_line = i
                break
        
        # Parse numeric data
        raw_data = []
        for line in motion_lines[data_start_line:]:
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            raw_data.append(vals)
        
        if not raw_data:
            return
        
        raw_data = np.array(raw_data)
        num_frames = raw_data.shape[0]
        num_bones = len(self.bones)
        
        self.positions = np.zeros((num_frames, 3))
        self.rotations = np.zeros((num_frames, num_bones, 3))
        
        col_idx = 0
        for (b_idx, ch_type) in self.channel_map:
            if col_idx >= raw_data.shape[1]:
                break
            
            vals = raw_data[:, col_idx]
            col_idx += 1
            
            # Position channels (root only, already in meters)
            if ch_type == 'Xposition':
                self.positions[:, 0] = vals
            elif ch_type == 'Yposition':
                self.positions[:, 1] = vals
            elif ch_type == 'Zposition':
                self.positions[:, 2] = vals
            # Rotation channels (stored by axis, not by order)
            elif ch_type == 'Xrotation':
                self.rotations[:, b_idx, 0] = vals
            elif ch_type == 'Yrotation':
                self.rotations[:, b_idx, 1] = vals
            elif ch_type == 'Zrotation':
                self.rotations[:, b_idx, 2] = vals
        
        print(f"  Parsed {num_frames} frames. Root Pos Range: {np.ptp(self.positions, axis=0)}")
    
    def get_rotation_values(self, frame_idx: int, bone_idx: int) -> List[float]:
        """
        Get rotation values for a bone at a specific frame, in the correct order.
        
        Args:
            frame_idx: Frame index
            bone_idx: Bone index
            
        Returns:
            List of rotation values in the order specified by self.orders[bone_idx]
        """
        order = self.orders[bone_idx] or "zxy"
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        return [self.rotations[frame_idx, bone_idx, axis_map[c]] for c in order]


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# =============================================================================

# These functions maintain backward compatibility with existing code

def read_bvh(filename: str, read_motion: bool = True) -> BVHParser:
    """
    Parse a BVH file and return a BVHParser object.
    
    Backward-compatible wrapper for BVHParser.
    """
    parser = BVHParser(filename)
    parser.parse(read_motion=read_motion)
    return parser

def read_bvh_hierarchy(filename: str) -> BVHParser:
    """Parse only the hierarchy of a BVH file."""
    return read_bvh(filename, read_motion=False)

def read_bvh_motion(filename: str, parser: BVHParser) -> BVHParser:
    """Parse motion data for an existing parser (re-parses the file)."""
    parser._parse_motion()
    return parser
