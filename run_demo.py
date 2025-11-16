"""Small runner script to exercise the Phase 1 scaffold.

Usage:
    python run_demo.py --config configs/sample_config.yaml
"""
import argparse
from cerberus.cli import run_from_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
