"""Small runner script for Project Cerberus.

Usage:
    # Run attacks and evaluation (Phase 1)
    python run_demo.py --mode eval --config configs/sample_config.yaml
    
    # Train models (Phase 2)
    python run_demo.py --mode train --training-type baseline --config configs/training_config.yaml
    python run_demo.py --mode train --training-type adversarial --config configs/training_config.yaml
"""
import argparse
from cerberus.cli import run_from_config, run_training


def main():
    p = argparse.ArgumentParser(
        description="Cerberus - Adversarial AI Simulation Framework"
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["eval", "train"],
        default="eval",
        help="Mode: 'eval' for attack/evaluation (Phase 1), 'train' for model training (Phase 2)"
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    p.add_argument(
        "--training-type",
        type=str,
        choices=["baseline", "adversarial"],
        help="Training type: 'baseline' or 'adversarial' (only used in train mode)"
    )
    p.add_argument(
        "--epochs",
        type=int,
        help="Override number of training epochs from config"
    )
    p.add_argument(
        "--output",
        type=str,
        help="Override output path for trained model"
    )
    
    args = p.parse_args()
    
    if args.mode == "eval":
        # Phase 1: Run attacks and evaluation
        run_from_config(args.config)
    elif args.mode == "train":
        # Phase 2: Train models
        if not args.training_type:
            print("❌ Error: --training-type is required for training mode")
            print("   Use: --training-type baseline|adversarial")
            return
        
        run_training(
            config_path=args.config,
            training_type=args.training_type,
            num_epochs=args.epochs,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
