# Project Cerberus — Adversarial AI Simulation Framework

[![CI](https://github.com/YOUR_USERNAME/major_projekt/a## 📋 Phase 1 Deliverables (Current)

- [x] Scaffold and package structure
- [x] Config loader (YAML)
- [x] PyTorch model loader
- [x] CIFAR-10 dataset integration
- [x] FGSM attack via ART
- [x] HTML report generation
- [x] Docker containerization (CPU)
- [x] Unit tests with mocks
- [x] CI/CD with GitHub Actions

##  Complete Documentation

This project includes comprehensive documentation:

-  **[README.md](README.md)** (this file) — Quick start and overview
-  **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** — Deep dive into what/why/how
-  **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Commands, troubleshooting, daily use
-  **[TIMELINE.md](TIMELINE.md)** — Project phases and roadmap
-  **[PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)** — Phase 1 achievements
-  **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** — Visual project overview
-  **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** — Documentation index

**New to the project?** Start with this README, then see [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for a guide to all docs.

##  Project Timeline

See [TIMELINE.md](TIMELINE.md) for the full phase breakdown and milestones.s/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/major_projekt/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An automated testing framework for evaluating and hardening AI models against adversarial attacks. Built for Dayananda Sagar University CSE Project Phase I (2025-2026).

##  Project Overview

Project Cerberus provides a robust, containerized pipeline to:
- Load pre-trained PyTorch models
- Execute adversarial attacks (FGSM, PGD, etc.) using IBM ART
- Apply defense mechanisms (adversarial retraining)
- Generate comprehensive evaluation reports

**Key Features:**
-  Pure Python with PyTorch and IBM Adversarial Robustness Toolbox (ART)
- 🐳 Fully containerized with Docker (CPU-only, no GPU required)
-  YAML-based configuration for reproducible experiments
-  HTML/PDF report generation with metrics and visualizations
-  Modular architecture with extensive unit tests

##  Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t cerberus-demo .

# Run the demo
docker run --rm -v $(pwd)/outputs:/app/outputs cerberus-demo
```

The demo will download CIFAR-10, run FGSM attack, and generate `outputs/report.html`.

### Option 2: Local Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install CPU-only PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install other dependencies
pip install -r requirements.txt

# Run the demo
python run_demo.py --config configs/sample_config.yaml
```

##  Requirements

- Python 3.9+
- Docker (for containerized runs)
- ~2GB disk space for dependencies and datasets

##  Running Tests

```bash
# Install test dependencies
pip install -r test_requirements.txt

# Run tests with coverage
pytest -v --cov=cerberus

# All tests use mocks, so torch/ART are NOT required for testing
```

## 📁 Project Structure

```
major_projekt/
├── cerberus/           # Main package
│   ├── __init__.py
│   ├── config.py       # YAML config loader
│   ├── model.py        # Model ingestion
│   ├── dataset.py      # Dataset loaders
│   ├── attacks.py      # Adversarial attack wrappers (ART)
│   ├── report.py       # Report generation
│   └── cli.py          # CLI pipeline orchestration
├── configs/            # Sample YAML configs
├── tests/              # Unit tests (with mocks)
├── run_demo.py         # Demo runner script
├── Dockerfile          # CPU-only container
├── requirements.txt    # Runtime dependencies
└── test_requirements.txt  # Test dependencies
```

## ⚙️ Configuration

Edit `configs/sample_config.yaml` to customize:

```yaml
model:
  path: null  # Path to .pt/.pth model, or null for demo model

dataset:
  name: cifar10
  root: ./data
  batch_size: 64

attack:
  name: fgsm
  eps: 0.03

output:
  report_path: outputs/report.html
```

##  Phase 1 Deliverables (Current)

- [x] Scaffold and package structure
- [x] Config loader (YAML)
- [x] PyTorch model loader
- [x] CIFAR-10 dataset integration
- [x] FGSM attack via ART
- [x] HTML report generation
- [x] Docker containerization (CPU)
- [x] Unit tests with mocks
- [x] CI/CD with GitHub Actions

##  Project Timeline

See [TIMELINE.md](TIMELINE.md) for the full phase breakdown and milestones.

##  Team

- Chhavi Sharma (ENG22CS0278)
- Gaurav Bhandare (ENG22CS0305)
- Chiranjeev Kapoor (ENG22CS0281)
- B Dheerendra Achar (ENG22CS0534)

**Supervisor:** Prof. Dharmendra D P  
**Batch:** 144 | **Department:** CSE, School of Engineering, Dayananda Sagar University

##  License

This project is part of an academic submission. All rights reserved.

##  Acknowledgments

- IBM Adversarial Robustness Toolbox (ART)
- PyTorch Team
