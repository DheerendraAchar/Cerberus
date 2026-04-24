# CERBERUS v2

CERBERUS is an open-source adversarial robustness platform for evaluating attacks, defenses, transferability, and retraining workflows across image and text models.

- Backend: Flask + PyTorch
- Frontend: React (MUI)
- Datasets: CIFAR-10, AG News
- Focus: fast experimentation + demo-friendly workflows + extensibility

## Highlights

- Rich attack suite for images and text
- Defense comparison flows for standard vs robust models
- Transferability analysis across architectures
- Retraining pipeline with stop control + demo retrain mode
- Experiment and retraining history tracking
- Modular codebase so users can consume only selected components (attacks/defense/retraining)

---

## Project Structure

```text
.
├── backend.py                    # Main Flask API server
├── frontend/                     # React app
│   ├── src/components/
│   └── package.json
├── cerberus/
│   ├── attacks/                  # Image + text attack implementations
│   ├── dataset.py                # CIFAR-10 and AG News loaders
│   ├── model.py
│   └── adversarial_training.py
├── scripts/                      # Repro scripts and utilities
├── configs/                      # YAML configs
├── tests/                        # Python tests
├── requirements.txt
└── LICENSE
```

---

## Supported Capabilities (Current)

### Datasets
- CIFAR-10 (image classification)
- AG News (text classification)

### Attack Coverage

#### Image attacks (CIFAR-10)
- FGSM
- PGD
- C&W
- DeepFool
- JSMA
- AutoAttack
- Square
- FAB
- RayS
- TRADES attack mode

#### Text attacks (AG News)
- FGSM-style token perturbation
- PGD-style iterative token perturbation
- TokenSwap
- TokenNoise
- TokenSubstitution

### Defense + Transfer
- Defense comparison: CIFAR-10 + AG News
- Transfer analysis: CIFAR-10 + AG News

### Retraining
- Auto-triggered retraining (configurable threshold path)
- Explicit stop control
- Demo retraining mode (fast CPU demo path)
- Retraining history with status + metrics

---

## Quick Start

### 1) Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch separately based on your environment:

```bash
# CPU example
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

### 2) Frontend

```bash
cd frontend
npm install
```

### 3) Run

Terminal 1 (backend):

```bash
source .venv/bin/activate
python backend.py
```

Terminal 2 (frontend):

```bash
cd frontend
npm start
```

Frontend: http://localhost:3000  
Backend API: http://localhost:5000

---

## API Surface (Core)

- `POST /api/run-attack`
- `POST /api/run-defense`
- `POST /api/transfer-analysis`
- `GET /api/retraining-status`
- `POST /api/retraining-stop`
- `GET|POST /api/retraining-config`
- `GET /api/retraining-history`
- `GET /api/experiments`

---

## Use Only Selective Modules

If you only need part of CERBERUS, you can import/use modules directly.

### Use only attack module

```python
from cerberus.attacks import FSGMAttack, PGDAttack
from cerberus.attacks.text_attacks import TokenSwapAttack
```

### Use only dataset module

```python
from cerberus.dataset import get_cifar10_loaders, get_ag_news_loaders
```

### Use only defense/transfer through API

Run backend and call:
- `/api/run-defense`
- `/api/transfer-analysis`

This allows partial adoption without running the full UI stack.

---

## Contributing

We welcome contributions for:
- new attacks
- new model architectures
- new retraining mechanisms
- robustness/evaluation improvements
- tests and docs

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an MR/PR.

### Contribution Workflow (MR/PR)

1. Fork repository
2. Create a feature branch (`feat/<short-name>`)
3. Implement changes with tests/docs
4. Run checks locally
5. Open a Merge Request / Pull Request
6. Address review comments
7. **Final merge happens only after approval by CERBERUS maintainers**

### Mandatory Rules

All MRs/PRs must:

1. **Preserve existing pipelines**
	- CIFAR workflows must not regress
	- AG News workflows must not regress
	- If changing shared logic, include regression checks

2. **Be modular**
	- New attack/model/retraining code should be isolated and reusable
	- No hardcoded dataset-specific hacks inside generic utilities

3. **Include tests**
	- Add or update tests for new behavior
	- Keep existing tests passing

4. **Document changes**
	- Update README and/or relevant docs for any user-visible behavior
	- Include usage examples for new attack/model/retraining mechanism

5. **Follow code quality standards**
	- Clear naming
	- Small focused functions
	- Backward compatible API behavior where possible

6. **No secrets / no large runtime artifacts**
	- Never commit local DBs, model checkpoints, datasets, env files, logs

### Recommended PR Template Checklist

- [ ] Added tests
- [ ] Updated docs
- [ ] Verified CIFAR path still works
- [ ] Verified AG News path still works
- [ ] Added migration notes (if breaking change)
- [ ] No generated runtime artifacts committed

---

## Extending CERBERUS

### Add a new attack
- Add implementation under `cerberus/attacks/`
- Export it in `cerberus/attacks/__init__.py`
- Register in backend dispatch map + UI dropdowns
- Add tests and README update

### Add a new model
- Add model construction path in backend/model module
- Add architecture option in UI
- Add loading/evaluation support for dataset(s)
- Add tests and benchmark notes

### Add a new retraining mechanism
- Add retrain mode/strategy logic in backend
- Wire status/progress/history fields
- Ensure stop handling remains cooperative
- Add tests + docs + rollback-safe defaults

---

## Development Checks

Backend syntax:

```bash
python -m py_compile backend.py
```

Python tests:

```bash
pip install -r test_requirements.txt
pytest -v
```

Frontend build:

```bash
cd frontend
npm run build
```

---

## Open Source Governance

- Maintainers: CERBERUS core maintainers
- Review gate: at least one maintainer approval
- Merge authority: CERBERUS maintainers only
- We may request design changes to preserve architecture consistency and research quality

---

## License

MIT License. See [LICENSE](LICENSE).
