"""Locations of the data files bundled with the package."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MAP = DATA_DIR / "map.yaml"
DEFAULT_SKILLS = DATA_DIR / "skills.json"
