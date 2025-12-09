import tomllib
from pathlib import Path


def load_config():
    config_path = Path(__file__).parent.parent / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)

