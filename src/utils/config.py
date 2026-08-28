"""Load the shared project config.yaml from anywhere in the codebase."""
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
