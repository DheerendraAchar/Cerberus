"""Config loader for YAML-based test configs."""
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return as dict.

    Lazy-imports PyYAML so top-level imports stay light.
    """
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required to load configuration files") from exc

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
