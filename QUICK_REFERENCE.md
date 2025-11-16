# Project Cerberus — Quick Reference Guide

**For:** Team members, supervisors, and future contributors  
**Version:** 0.1.0 (Phase 1)

---

## 🚀 Quick Start Commands

### Run Demo (Docker - Recommended)
```bash
# Build image (one-time, ~17 min)
docker build -t cerberus-demo .

# Run demo
docker run --rm -v $(pwd)/outputs:/app/outputs cerberus-demo

# View report
open outputs/report.html
```

### Run Demo (Local)
```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt

# Run
python run_demo.py --config configs/sample_config.yaml

# View report
open outputs/report.html
```

### Run Tests
```bash
# Install test deps
pip install -r test_requirements.txt

# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=cerberus --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## 📝 Common Tasks

### Create a New Experiment Config

```yaml
# configs/my_experiment.yaml
model:
  path: /path/to/your/model.pt  # or null for demo

dataset:
  name: cifar10
  root: ./data
  batch_size: 128
  num_workers: 2

attack:
  name: fgsm
  eps: 0.05  # Try different epsilon values

output:
  report_path: outputs/my_experiment_report.html
```

Run with:
```bash
python run_demo.py --config configs/my_experiment.yaml
```

---

### Load Your Own Model

**Option 1: PyTorch Checkpoint**
```python
# Save your model
import torch
torch.save(model, "my_model.pt")
```

Update config:
```yaml
model:
  path: my_model.pt
```

**Option 2: State Dict**
```python
# Save state dict
torch.save(model.state_dict(), "my_model_state.pth")
```

You'll need to modify `cerberus/model.py` to load your specific architecture.

---

### Add a New Attack

**Step 1:** Add function to `cerberus/attacks.py`
```python
def run_pgd_attack(model, test_loader, eps=0.03, device="cpu"):
    """Run PGD attack."""
    from art.attacks.evasion import ProjectedGradientDescent
    from art.estimators.classification import PyTorchClassifier
    
    # Similar structure to run_fgsm_attack
    # ...
```

**Step 2:** Update `cerberus/cli.py`
```python
if attack_cfg.get("name") == "pgd":
    from .attacks import run_pgd_attack
    res = run_pgd_attack(model, test_loader, ...)
```

**Step 3:** Update config
```yaml
attack:
  name: pgd
  eps: 0.03
  iterations: 10
```

---

### View Test Coverage

```bash
# Generate coverage report
pytest --cov=cerberus --cov-report=html

# Open in browser
open htmlcov/index.html
```

---

## 🐛 Troubleshooting

### Issue: Docker build fails downloading PyTorch

**Error:**
```
ReadTimeoutError: Read timed out
```

**Solutions:**
1. Retry build (network issue):
   ```bash
   docker build -t cerberus-demo .
   ```

2. Increase Docker memory:
   - Docker Desktop → Preferences → Resources → Memory: 4GB+

3. Use pre-built image (future):
   ```bash
   docker pull ghcr.io/your-username/cerberus:latest
   ```

---

### Issue: "torch" not found when running locally

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
# Install CPU-only PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

**macOS with Apple Silicon (M1/M2):**
```bash
# Use native macOS wheels
pip install torch torchvision
```

---

### Issue: Tests fail with import errors

**Error:**
```
ModuleNotFoundError: No module named 'yaml'
```

**Solution:**
```bash
# Install test requirements
pip install -r test_requirements.txt
```

Note: Tests use mocks for torch/ART, but still need PyYAML and Jinja2.

---

### Issue: Out of memory during attack

**Error:**
```
RuntimeError: [enforce fail at alloc_cpu.cpp:...] out of memory
```

**Solution 1:** Reduce batch size in config:
```yaml
dataset:
  batch_size: 32  # Try smaller values like 16, 8
```

**Solution 2:** Process in smaller chunks:
Edit `cerberus/attacks.py`, line ~75:
```python
batch = 128  # Reduce this value
```

---

### Issue: CIFAR-10 download hangs

**Error:**
```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz... (hangs)
```

**Solution 1:** Wait (can take 5-10 minutes on slow connections)

**Solution 2:** Download manually:
```bash
mkdir -p data/cifar-10-batches-py
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf cifar-10-python.tar.gz
```

---

### Issue: Report not generated

**Error:** No error, but `outputs/report.html` doesn't exist.

**Solution 1:** Check permissions:
```bash
mkdir -p outputs
chmod 755 outputs
```

**Solution 2:** Check for errors in logs:
```bash
python run_demo.py --config configs/sample_config.yaml 2>&1 | tee run.log
cat run.log
```

---

## 📊 Interpreting Results

### Understanding Report Metrics

**Baseline Accuracy:**
- Model performance on clean data
- Should match published accuracy for pre-trained models
- Low baseline (<50%) indicates untrained or broken model

**Adversarial Accuracy:**
- Model performance on perturbed data
- Lower = more vulnerable to attacks
- Rule of thumb:
  - >80%: Robust model
  - 50-80%: Moderately vulnerable
  - <50%: Highly vulnerable

**Attack Success Rate:**
```
Success Rate = (Baseline Acc - Adv Acc) / Baseline Acc × 100%
```

**Example:**
- Baseline: 92%
- Adversarial: 45%
- Success Rate: (92-45)/92 = 51% of predictions flipped

---

## 🔧 Configuration Options

### Complete Config Schema

```yaml
model:
  path: string | null          # Path to model file or null for demo
  
dataset:
  name: string                 # "cifar10" (more coming in Phase 2)
  root: string                 # Path to dataset cache
  batch_size: integer          # Batch size for DataLoader (default: 64)
  num_workers: integer         # DataLoader workers (default: 0)

attack:
  name: string                 # "fgsm" (more in Phase 2/3)
  eps: float                   # Perturbation magnitude (0.0-1.0)

output:
  report_path: string          # Path to save HTML report
```

---

## 🧪 Development Workflow

### Making Changes

```bash
# 1. Create branch
git checkout -b feature/my-feature

# 2. Make changes
# Edit cerberus/*.py

# 3. Run tests
pytest -v

# 4. Check formatting
black cerberus tests
isort cerberus tests
flake8 cerberus tests --max-line-length=120

# 5. Commit
git add .
git commit -m "Add feature: ..."

# 6. Push and create PR
git push origin feature/my-feature
```

### Adding Dependencies

**Runtime dependency:**
```bash
# Add to requirements.txt
echo "new-package>=1.0" >> requirements.txt

# Update Docker image
docker build -t cerberus-demo .
```

**Test dependency:**
```bash
# Add to test_requirements.txt
echo "new-test-package>=1.0" >> test_requirements.txt

# Install
pip install -r test_requirements.txt
```

---

## 📚 File Quick Reference

| File | Purpose |
|------|---------|
| `cerberus/config.py` | Load YAML configs |
| `cerberus/model.py` | Load PyTorch models |
| `cerberus/dataset.py` | Load datasets (CIFAR-10) |
| `cerberus/attacks.py` | Execute attacks (FGSM) |
| `cerberus/report.py` | Generate HTML reports |
| `cerberus/cli.py` | Orchestrate pipeline |
| `run_demo.py` | CLI entry point |
| `configs/sample_config.yaml` | Example config |
| `tests/test_*.py` | Unit tests |
| `Dockerfile` | Container definition |
| `requirements.txt` | Runtime dependencies |
| `test_requirements.txt` | Test dependencies |
| `README.md` | User guide |
| `TIMELINE.md` | Project phases |
| `TECHNICAL_DOCUMENTATION.md` | What/Why/How guide |

---

## 🎯 Performance Benchmarks

**Tested on MacBook Pro (Intel i7, 16GB RAM, No GPU):**

| Task | Time | Notes |
|------|------|-------|
| Docker build (first time) | ~17 min | One-time cost |
| Docker build (code change) | <1 min | Cached layers |
| CIFAR-10 download | 3-5 min | One-time, cached |
| FGSM on 10K images (untrained model) | 2-3 min | CPU-only |
| Test suite | <1 sec | Mocked dependencies |
| Report generation | <0.1 sec | Simple HTML |

**Expected on GPU:**
- FGSM attack: ~10-30 sec (10-20x faster)

---

## 🆘 Getting Help

### Resources

1. **README.md** — Start here for quick start
2. **TECHNICAL_DOCUMENTATION.md** — Deep dive into implementation
3. **TIMELINE.md** — Project phases and roadmap
4. **This file** — Quick commands and troubleshooting

### Contact

**Supervisor:** Prof. Dharmendra D P  
**Team Lead:** [Add email]  
**Repository Issues:** [GitHub Issues URL]

### Common Questions

**Q: Can I run this on GPU?**  
A: Phase 1 is CPU-only. GPU support planned for Phase 4.

**Q: How do I add my own dataset?**  
A: Create a loader function in `cerberus/dataset.py` following the CIFAR-10 pattern.

**Q: Can I use TensorFlow models?**  
A: Not yet. Phase 3 may add TensorFlow support via ART's TensorFlowV2Classifier.

**Q: How accurate are the demo results?**  
A: The demo TinyCNN is untrained (random weights), so accuracies are ~10%. Use a pre-trained model for meaningful results.

**Q: Can I export reports to PDF?**  
A: Not in Phase 1. Phase 2 will add PDF export via WeasyPrint.

---

**Last Updated:** November 16, 2025  
**Maintained By:** Project Cerberus Team
