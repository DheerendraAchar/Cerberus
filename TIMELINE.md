# Project Cerberus — Timeline & Milestones

**Project Duration:** Academic Year 2025-2026 (Phase I)  
**Team:** Batch 144 | CSE, Dayananda Sagar University

---

##  Phase Breakdown

### Phase 0: Planning & Setup ✅ (Completed: Nov 2025)

**Duration:** 0.5–1 day  
**Status:** ✅ COMPLETED

**Deliverables:**
- [x] Proposal reviewed and requirements extracted
- [x] Technology stack finalized (Python, PyTorch, ART, Docker)
- [x] Development environment set up
- [x] Git repository initialized

---

### Phase 1: MVP & Core Framework ✅ (Completed: Nov 2025)

**Duration:** 4–7 days  
**Status:** ✅ COMPLETED

**Objectives:**
- Build minimal viable product demonstrating end-to-end adversarial testing pipeline
- Establish code quality standards with tests and CI

**Deliverables:**
- [x] Python package structure (`cerberus/`)
- [x] YAML configuration system
- [x] PyTorch model loader (`.pt`, `.pth` support)
- [x] CIFAR-10 dataset integration
- [x] FGSM attack implementation via ART
- [x] Basic HTML report generation
- [x] CPU-only Dockerfile
- [x] Unit tests (8 tests, 100% pass rate)
- [x] GitHub Actions CI pipeline
- [x] Documentation (README, inline docstrings)

**Demo Capability:**  
Run `docker run cerberus-demo` to execute FGSM on CIFAR-10 and generate report.

---

### Phase 2: Defenses & Advanced Reporting (Planned: Dec 2025)

**Duration:** 4–6 days  
**Status:** 🔄 NOT STARTED

**Objectives:**
- Implement defense mechanisms
- Enhance reporting with visualizations and comparative analysis

**Planned Deliverables:**
- [ ] Adversarial retraining pipeline
- [ ] Additional defense: Input transformation / Feature squeezing
- [ ] Enhanced reports:
  - [ ] Accuracy plots (baseline vs. post-attack vs. post-defense)
  - [ ] Sample adversarial example images
  - [ ] PDF export support
- [ ] Model serialization (save hardened models)
- [ ] Integration tests for full pipeline
- [ ] Performance benchmarking

**Demo Capability:**  
Run attack → apply defense → retrain → compare metrics in visual report.

---

### Phase 3: Extensibility & Multi-Domain Support (Planned: Jan 2026)

**Duration:** 5–8 days  
**Status:** 🔄 NOT STARTED

**Objectives:**
- Support multiple attack types and domains (vision + NLP)
- Plugin architecture for easy extension

**Planned Deliverables:**
- [ ] Additional attacks:
  - [ ] PGD (Projected Gradient Descent)
  - [ ] C&W (Carlini & Wagner)
  - [ ] DeepFool
- [ ] NLP pipeline:
  - [ ] Text dataset loaders (IMDB, SST)
  - [ ] Text attack wrappers (TextFooler, etc.)
- [ ] Plugin system:
  - [ ] Attack plugin interface
  - [ ] Defense plugin interface
  - [ ] Dataset loader plugins
- [ ] CLI enhancements (subcommands, verbose logging)
- [ ] Configuration validation and error handling

**Demo Capability:**  
Run multiple attacks on both vision and NLP models with custom plugins.

---

### Phase 4: Final Deliverables & Evaluation (Planned: Feb 2026)

**Duration:** 3–5 days  
**Status:** 🔄 NOT STARTED

**Objectives:**
- Complete all documentation and evaluation artifacts
- Prepare for final submission and presentation

**Planned Deliverables:**
- [ ] Comprehensive experiments:
  - [ ] Benchmark on multiple datasets (CIFAR-10, MNIST, ImageNet subset)
  - [ ] Attack success rate analysis
  - [ ] Defense effectiveness metrics
- [ ] Final project report (PDF)
- [ ] Presentation slides
- [ ] User guide and API documentation
- [ ] Published Docker image (DockerHub or GitHub Packages)
- [ ] Demo video (optional)
- [ ] Code quality review:
  - [ ] Linting (black, flake8, isort)
  - [ ] Type hints (mypy)
  - [ ] Coverage target: >80%

**Demo Capability:**  
Complete, production-ready framework ready for academic submission and potential real-world use.

---

##  Progress Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 0: Planning | ✅ | 100% |
| Phase 1: MVP | ✅ | 100% |
| Phase 2: Defenses | 🔄 | 0% |
| Phase 3: Extensibility | 🔄 | 0% |
| Phase 4: Final Deliverables | 🔄 | 0% |

**Overall Progress:** 40% (2/5 phases complete)

---

##  Next Immediate Steps

1. **Implement adversarial retraining** (Phase 2)
2. **Add visualization to reports** (plots, sample images)
3. **Integrate PGD attack** (Phase 3 prep)
4. **Create sample NLP pipeline** (Phase 3)

---

## ⚠️ Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large model OOM on CPU | High | Add batch processing, memory profiling, early warnings |
| ART compatibility issues | Medium | Pin ART version, test on multiple torch versions |
| Dataset download failures | Low | Cache datasets, provide offline mode |
| Time constraints | High | Prioritize core features, defer nice-to-haves |

---

## 📞 Contact & Support

**Supervisor:** Prof. Dharmendra D P  
**Team Lead:** dheerudivya0408@gmail.com 
**Repository:** https://github.com/DheerendraAchar/Cerberus

---

*Last Updated: November 16, 2025*
