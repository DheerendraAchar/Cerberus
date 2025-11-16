"""Tests for config loader."""
import tempfile
import os


def test_load_config():
    """Test loading a simple YAML config."""
    from cerberus.config import load_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("model:\n  path: /tmp/model.pt\ndataset:\n  name: cifar10\n")
        f.flush()
        path = f.name

    try:
        cfg = load_config(path)
        assert "model" in cfg
        assert cfg["model"]["path"] == "/tmp/model.pt"
        assert cfg["dataset"]["name"] == "cifar10"
    finally:
        os.unlink(path)
