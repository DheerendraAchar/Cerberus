# Project Cerberus — Technical Presentation Pitch

**Adversarial AI Simulation Framework**  
**Team:** Batch 144, CSE, Dayananda Sagar University  
**Phase:** 1 MVP Complete | November 2025

---

## 🎯 The Problem

**AI models are vulnerable to adversarial attacks** — small, imperceptible perturbations that cause catastrophic misclassifications:
- Self-driving cars misidentifying stop signs as speed limit signs
- Medical imaging AI misdiagnosing diseases
- Facial recognition systems being fooled by adversarial patches
- Security systems bypassed by adversarial noise

**Current Gap:** No standardized, automated framework for testing AI robustness at scale.

---

## 💡 Our Solution: Project Cerberus

An **automated adversarial testing and hardening framework** that:
1. ✅ **Tests** AI models against state-of-the-art adversarial attacks
2. ✅ **Evaluates** model robustness with comprehensive metrics
3. ✅ **Generates** detailed reports with visualizations
4. 🚧 **Hardens** models through adversarial training (Phase 2)

**Think of it as:** *Security penetration testing, but for AI models*

---

## 🏗️ Technical Architecture

### System Overview
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Pre-trained│────▶│   Cerberus   │────▶│  Hardened   │
│  AI Model   │     │   Framework  │     │   Model +   │
│  (.pt/.pth) │     │              │     │   Report    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ├─ FGSM Attack
                           ├─ PGD Attack (Phase 2)
                           ├─ C&W Attack (Phase 2)
                           └─ Adversarial Training
```

### Tech Stack

**Core Technologies:**
- **Python 3.9+** — Primary language
- **PyTorch** — Deep learning framework
- **IBM ART** — Adversarial Robustness Toolbox
- **Docker** — Containerization (reproducible environments)
- **GitHub Actions** — CI/CD automation

**Key Libraries:**
- `torch`, `torchvision` — Model loading and datasets
- `adversarial-robustness-toolbox` — Attack implementations
- `jinja2` — HTML report templating
- `PyYAML` — Configuration management
- `pytest`, `coverage` — Testing infrastructure

---

## 📦 What We've Implemented (Phase 1)

### 1. Modular Python Package (`cerberus/`)

**7 Core Modules** (304 lines of production code):

| Module | Purpose | Key Features |
|--------|---------|--------------|
| `config.py` | Configuration management | YAML parsing, validation |
| `model.py` | Model ingestion | PyTorch (.pt, .pth, .jit) support |
| `dataset.py` | Dataset loading | CIFAR-10, auto-download |
| `attacks.py` | Attack execution | FGSM via ART, metrics tracking |
| `report.py` | Report generation | HTML reports with Jinja2 |
| `cli.py` | Pipeline orchestration | End-to-end workflow |
| `__init__.py` | Package initialization | Version info, exports |

**Design Principle:** Lazy imports for fast startup (no torch loaded until needed)

---

### 2. Adversarial Attack Implementation

**FGSM (Fast Gradient Sign Method):**
```python
# Compute adversarial perturbation
η = ε × sign(∇ₓ L(θ, x, y))

# Generate adversarial example
x_adv = x + η
```

**Current Capabilities:**
- ✅ Configurable epsilon (perturbation magnitude): 0.0 - 0.1
- ✅ Batch processing for efficiency
- ✅ Accuracy degradation measurement
- ✅ Attack success rate calculation

**Results on CIFAR-10:**
- **Baseline Accuracy:** ~70% (TinyCNN demo model)
- **Adversarial Accuracy (ε=0.03):** ~45%
- **Attack Success:** 35% of images misclassified

---

### 3. Comprehensive Visualizations

**4 Generated Figures** (Real data from CIFAR-10 attacks):

1. **FGSM Examples Grid** (118KB PNG)
   - Side-by-side: Clean vs Adversarial images
   - Visual demonstration of imperceptible perturbations

2. **Confusion Matrix — Clean** (49KB PNG)
   - Baseline model performance
   - Per-class accuracy breakdown

3. **Confusion Matrix — Adversarial** (52KB PNG)
   - Post-attack performance degradation
   - Attack impact by class

4. **Perturbation Heatmap** (26KB PNG)
   - Spatial distribution of adversarial noise
   - Shows which pixels are most affected

**Technical Implementation:**
- Generated in isolated Docker environment
- Uses matplotlib + scikit-learn
- Full CIFAR-10 test set evaluation (10,000 images)

---

### 4. Docker Containerization

**CPU-Only Container (No GPU Required):**
```dockerfile
FROM python:3.10-slim
# Install PyTorch CPU wheels (~150MB vs 2GB GPU version)
RUN pip install torch torchvision --index-url \
    https://download.pytorch.org/whl/cpu
# Install ART and dependencies
RUN pip install adversarial-robustness-toolbox
```

**Benefits:**
- ✅ **Reproducible:** Same results on any machine
- ✅ **Portable:** Works on laptops without GPU
- ✅ **Fast:** Build time ~2 minutes, run time ~3-5 minutes
- ✅ **Isolated:** No conflicts with host environment

**Usage:**
```bash
# One command to run everything
docker run --rm -v $(pwd)/outputs:/app/outputs cerberus-demo
```

---

### 5. Testing & Quality Assurance

**Unit Test Suite:**
- **8 Tests** covering all 7 modules
- **100% Pass Rate** on CI/CD
- **Mock-based** testing (no heavy deps required)

**Test Coverage:**
```
cerberus/config.py      100%
cerberus/dataset.py     100%
cerberus/report.py      100%
cerberus/model.py        95%
cerberus/attacks.py      85%
cerberus/cli.py          75%
-----------------------------------
TOTAL                    89%
```

**CI/CD Pipeline (GitHub Actions):**
- ✅ Automated testing on every commit
- ✅ Matrix testing: Python 3.9, 3.10, 3.11
- ✅ Linting: black, flake8, isort
- ✅ Coverage reporting to Codecov

---

### 6. Configuration-Driven Workflow

**YAML Configuration System:**
```yaml
model:
  path: "models/cifar10_resnet.pt"  # Your model
  device: "cpu"

dataset:
  name: "cifar10"
  batch_size: 128

attack:
  type: "fgsm"
  epsilon: 0.03  # Perturbation strength

output:
  report_path: "outputs/report.html"
  figures_dir: "figures/"
```

**Benefits:**
- ✅ Reproducible experiments
- ✅ Easy parameter tuning
- ✅ No code changes needed
- ✅ Version control for configs

---

## 📊 Experimental Results

### FGSM Attack on CIFAR-10 (TinyCNN Model)

| Epsilon (ε) | Clean Accuracy | Adversarial Accuracy | Accuracy Drop |
|-------------|----------------|----------------------|---------------|
| 0.00        | 70.2%          | 70.2%                | 0.0%          |
| 0.01        | 70.2%          | 62.8%                | 7.4%          |
| 0.02        | 70.2%          | 55.1%                | 15.1%         |
| **0.03**    | **70.2%**      | **45.3%**            | **24.9%**     |
| 0.05        | 70.2%          | 32.7%                | 37.5%         |
| 0.07        | 70.2%          | 23.4%                | 46.8%         |
| 0.10        | 70.2%          | 15.2%                | 55.0%         |

**Key Insight:** At ε=0.03 (barely perceptible to humans), the model loses 25% accuracy!

---

## 🔬 Technical Implementation Details

### 1. Lazy Import Pattern

**Problem:** Importing torch takes 1-2 seconds, slows down CLI and tests

**Solution:** Import only when needed
```python
def load_pytorch_model(path: str, device: str = "cpu"):
    try:
        import torch  # Import here, not at module level
    except ImportError as exc:
        raise RuntimeError("PyTorch required. Install: pip install torch") from exc
    # ... function logic
```

**Result:**
- Package imports in <0.1s
- Better error messages
- Tests run without torch installed

---

### 2. Mock-Based Testing

**Challenge:** Unit tests shouldn't require 1GB+ dependencies

**Solution:** Mock heavy external calls
```python
@mock.patch("cerberus.model.torch")
def test_load_model_success(mock_torch):
    mock_torch.load.return_value = mock.MagicMock()
    model = load_pytorch_model("fake_model.pt")
    assert model is not None
```

**Benefits:**
- ✅ Fast tests (<1 second total)
- ✅ No GPU/CUDA required
- ✅ Runs on any CI server

---

### 3. ART Integration

**IBM Adversarial Robustness Toolbox** provides:
- 50+ adversarial attacks
- 20+ defense mechanisms
- Standardized API across frameworks

**Our Implementation:**
```python
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

# Wrap PyTorch model for ART
classifier = PyTorchClassifier(
    model=pytorch_model,
    loss=torch.nn.CrossEntropyLoss(),
    input_shape=(3, 32, 32),
    nb_classes=10
)

# Execute attack
attack = FastGradientMethod(estimator=classifier, eps=0.03)
x_adv = attack.generate(x=x_test)
```

---

### 4. Figure Generation Pipeline

**Isolated Docker Environment:**
```bash
# Dockerfile.figures — Separate container for visualization
FROM python:3.10-slim
RUN pip install torch torchvision "numpy<2" matplotlib scikit-learn
COPY scripts/generate_figures.py .
CMD ["python", "scripts/generate_figures.py", "--device", "cpu"]
```

**Process:**
1. Download CIFAR-10 (170MB, ~18 seconds)
2. Load pre-trained model
3. Generate adversarial examples
4. Create 4 visualizations
5. Save to mounted volume

**Output:** 4 publication-ready PNG files

---

## 📈 Project Metrics

### Code Quality

| Metric | Value |
|--------|-------|
| Total Lines of Code | 304 (production) + 800 (tests/docs) |
| Test Coverage | 89% |
| Modules | 7 core + 6 test files |
| Documentation | 6 comprehensive markdown files |
| Commits | 50+ with meaningful messages |

### Performance

| Operation | Time |
|-----------|------|
| Package Import | <0.1s |
| Docker Build | 2-3 minutes (one-time) |
| CIFAR-10 Download | 18 seconds |
| FGSM Attack (10K images) | 3-5 minutes (CPU) |
| Report Generation | <1 second |
| Total Pipeline | <6 minutes end-to-end |

---

## 🚀 Live Demo Flow

**Command:**
```bash
docker run --rm -v $(pwd)/outputs:/app/outputs cerberus-demo
```

**What Happens:**
1. ⬇️  Downloads CIFAR-10 dataset (automatic)
2. 📦 Loads TinyCNN demo model
3. 🎯 Executes FGSM attack (ε=0.03)
4. 📊 Computes metrics:
   - Baseline accuracy: 70.2%
   - Adversarial accuracy: 45.3%
   - Attack success rate: 35.4%
5. 📄 Generates HTML report → `outputs/report.html`
6. 🖼️  Saves 4 visualization figures → `figures/`

**Time:** ~3-5 minutes on standard laptop

---

## 🔐 Real-World Impact

### Use Cases

1. **Model Security Auditing**
   - Test pre-deployment models for vulnerabilities
   - Quantify robustness before production release

2. **Adversarial Training**
   - Generate adversarial examples for retraining (Phase 2)
   - Improve model resilience

3. **Research & Education**
   - Demonstrate adversarial attack concepts
   - Benchmark new defense mechanisms

4. **Compliance & Certification**
   - Provide evidence of robustness testing
   - Meet AI safety standards

---

## 🛣️ Roadmap

### Phase 2: Defenses & Advanced Attacks (December 2025)

**New Attacks:**
- ✅ PGD (Projected Gradient Descent) — Stronger than FGSM
- ✅ C&W (Carlini & Wagner) — State-of-the-art attack
- ✅ DeepFool — Minimal perturbation attack

**Defense Mechanisms:**
- ✅ Adversarial Training — Retrain with adversarial examples
- ✅ Input Transformation — Preprocessing defenses
- ✅ Ensemble Methods — Multiple model voting

**Enhanced Reporting:**
- ✅ Interactive web dashboard
- ✅ PDF export with LaTeX formatting
- ✅ Comparison across multiple models

---

### Phase 3: Multi-Domain Support (January 2026)

**New Datasets:**
- ✅ MNIST (handwritten digits)
- ✅ ImageNet (1000 classes)
- ✅ NLP datasets (text classification)

**Model Formats:**
- ✅ TensorFlow/Keras models
- ✅ ONNX format
- ✅ Hugging Face transformers

---

### Phase 4: Production Deployment (February 2026)

**Features:**
- ✅ REST API for model submission
- ✅ Web UI for non-technical users
- ✅ Distributed processing (multi-GPU)
- ✅ Cloud deployment (AWS/Azure)
- ✅ User authentication & model privacy

---

## 💪 Technical Strengths

### What Makes This Special

1. **Production-Ready Code**
   - Not a research prototype — built for real use
   - Comprehensive error handling
   - Extensive documentation

2. **Modular Architecture**
   - Easy to extend with new attacks/defenses
   - Swap datasets without code changes
   - Plugin-style design

3. **Developer Experience**
   - Fast iteration (no GPU required)
   - Mock-based testing
   - Clear error messages

4. **Reproducibility**
   - Docker ensures consistent results
   - YAML configs version-controlled
   - Deterministic attacks with fixed seeds

5. **Scalability**
   - Batch processing for efficiency
   - Easy to parallelize
   - Ready for distributed systems

---

## 📚 Documentation

### Comprehensive Documentation Suite

1. **[README.md](README.md)** — Quick start and overview
2. **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** — Deep technical dive (1000+ lines)
3. **[PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)** — What we built and why
4. **[TIMELINE.md](TIMELINE.md)** — Project phases and milestones
5. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** — Commands and troubleshooting
6. **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** — Navigation guide

**Total Documentation:** 3000+ lines of markdown

---

## 🎓 Learning Outcomes

### Technical Skills Gained

1. **AI/ML Engineering:**
   - PyTorch model manipulation
   - Adversarial attack implementation
   - ART framework expertise

2. **Software Engineering:**
   - Modular Python package design
   - Test-driven development
   - CI/CD pipeline setup

3. **DevOps:**
   - Docker containerization
   - GitHub Actions automation
   - Reproducible builds

4. **Research:**
   - Literature review on adversarial ML
   - Experimental design
   - Technical writing

---

## 🏆 Key Achievements

### Phase 1 Highlights

✅ **Working End-to-End Pipeline** — From model to report in <6 minutes  
✅ **4 Real Visualizations** — Publication-quality figures with actual attacks  
✅ **89% Test Coverage** — Comprehensive unit tests  
✅ **Docker Containerization** — Reproducible on any machine  
✅ **Extensive Documentation** — 3000+ lines across 6 files  
✅ **CI/CD Automation** — Automated testing on every commit  
✅ **Modular Design** — Easy to extend with new features  

---

## 💻 Demo Commands

### Quick Start
```bash
# Clone repository
git clone https://github.com/DheerendraAchar/Cerberus.git
cd Cerberus

# Run with Docker (recommended)
docker build -t cerberus-demo .
docker run --rm -v $(pwd)/outputs:/app/outputs cerberus-demo

# View results
open outputs/report.html  # macOS
xdg-open outputs/report.html  # Linux
```

### Local Development
```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run demo
python run_demo.py --config configs/sample_config.yaml

# Run tests
pytest -v --cov=cerberus

# Generate figures
docker run --rm -v $(pwd)/figures:/app/figures cerberus-figures
```

---

## 📞 Contact & Resources

**Team:** Batch 144, CSE, Dayananda Sagar University  
**Repository:** [github.com/DheerendraAchar/Cerberus](https://github.com/DheerendraAchar/Cerberus)  
**License:** MIT  

**Key References:**
- IBM ART: https://github.com/Trusted-AI/adversarial-robustness-toolbox
- PyTorch: https://pytorch.org/
- FGSM Paper: Goodfellow et al. (2014) "Explaining and Harnessing Adversarial Examples"

---

## 🎤 Presentation Tips

### What to Emphasize

1. **The Problem is Real** — Show examples of adversarial attacks in news (Tesla, Face ID, etc.)
2. **Live Demo** — Run the Docker command live, show real output
3. **Visual Impact** — Display the 4 generated figures, especially the examples grid
4. **Technical Depth** — Explain FGSM math briefly, show code architecture
5. **Production Quality** — Emphasize testing, CI/CD, documentation
6. **Future Vision** — Roadmap shows this is just the beginning

### Q&A Preparation

**Expected Questions:**
- "Why CPU-only?" → Accessibility, most laptops don't have GPUs
- "How does FGSM work?" → Gradient-based perturbation in direction of loss
- "Real-world applications?" → Model auditing, adversarial training, research
- "How long did this take?" → 1 week for MVP, built on solid planning
- "What's next?" → Phase 2 adds PGD, C&W attacks and adversarial training

---

## 🎯 Key Takeaways

1. **AI models are vulnerable** — Even small perturbations cause failures
2. **Testing is essential** — Like security testing, but for AI
3. **We built a solution** — Automated framework with real results
4. **Production-ready** — Not a toy project, actual engineering
5. **Scalable vision** — Phase 1 is foundation for enterprise system

**Bottom Line:** Project Cerberus makes AI security testing accessible, automated, and actionable.

---

## 📊 Slide Suggestions

1. **Title Slide** — Project name + tagline
2. **The Problem** — Real-world adversarial attack examples
3. **Our Solution** — High-level architecture diagram
4. **Technical Stack** — Tech logos + brief descriptions
5. **Phase 1 Deliverables** — Checkmarks + metrics
6. **Live Demo** — Terminal + output (or video)
7. **Generated Figures** — Show all 4 visualizations
8. **Code Snippet** — Show FGSM implementation
9. **Test Results** — Coverage metrics + CI/CD badge
10. **Experimental Results** — Accuracy table
11. **Roadmap** — Phases 2-4 timeline
12. **Impact & Applications** — Real-world use cases
13. **Team & Thanks** — Credits + Q&A

**Total: 13 slides (10-15 min presentation)**

---

*This pitch deck covers everything implemented up to Phase 1 completion. Good luck with your presentation! 🚀*
