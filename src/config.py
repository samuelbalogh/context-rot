import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
ALL_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
ENV_PATH = ALL_ROOT / ".env"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_env(env_path: Path | None = None):
    project_root = Path(__file__).parent.parent
    override = os.environ.get("CONTEXT_ROT_ENV")
    if override:
        load_dotenv(override)
    elif (project_root / ".env").exists():
        load_dotenv(project_root / ".env")
    else:
        load_dotenv(ENV_PATH)
    if env_path:
        load_dotenv(env_path)
