"""
Configuration loader for the Water Measuring system.

Reads config.yaml and provides typed access to all settings.
"""

from __future__ import annotations

import os
import yaml  # type: ignore


_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None) -> dict:
    """Load and return the configuration dictionary from a YAML file."""
    path = path or _DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# region Convenience accessors

def camera_cfg(cfg: dict, name: str) -> dict:
    """Return config dict for a specific camera ('top' or 'side')."""
    return cfg["cameras"][name]


def color_range(cfg: dict):
    """Return (lower, upper) numpy-ready lists for cv2.inRange."""
    return cfg["color"]["lower"], cfg["color"]["upper"]


def analysis_cfg(cfg: dict) -> dict:
    return cfg["analysis"]


def solenoid_cfg(cfg: dict) -> dict:
    return cfg["solenoid"]


def schedule_cfg(cfg: dict) -> dict:
    return cfg["schedule"]


def recording_cfg(cfg: dict) -> dict:
    return cfg["recording"]


def output_dir(cfg: dict, cam_name: str | None = None) -> str:
    """
    Return the output directory.  If cam_name is provided, returns a
    sub-directory for that camera (e.g. results/top/).
    """
    base = cfg["output"]["base_dir"]
    if cam_name:
        return os.path.join(base, cam_name)
    return base
# endregion
