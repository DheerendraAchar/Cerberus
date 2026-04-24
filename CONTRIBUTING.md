# Contributing to CERBERUS

Thanks for helping improve CERBERUS.

## Scope of Contributions

We welcome contributions for:
- new attacks
- new model architectures
- new retraining mechanisms
- evaluation/benchmark improvements
- bug fixes and documentation improvements

## Workflow

1. Fork the repository
2. Create a feature branch
   - `feat/<short-name>` for features
   - `fix/<short-name>` for bug fixes
3. Implement your changes
4. Add/Update tests
5. Update docs (README and relevant module docs)
6. Open a Merge Request / Pull Request
7. Address review comments
8. Merge only after maintainer approval

## Mandatory Rules

All contributions must:

1. Preserve existing behavior for both datasets:
   - CIFAR-10
   - AG News

2. Be modular:
   - keep attack/model/retraining logic isolated
   - avoid hard-coded assumptions in shared utilities

3. Include tests:
   - new functionality requires test coverage
   - no regressions in existing tests

4. Include docs:
   - usage notes for any new attack/model/retraining mechanism

5. Avoid committing local runtime artifacts:
   - datasets
   - model checkpoints
   - database files
   - logs
   - environment secrets

## Review & Merge Policy

- Maintainer review is required.
- Final merge authority belongs to CERBERUS maintainers.
- Maintainers may request architectural changes before approval.

## Local Validation Checklist

Before opening MR/PR:

- [ ] Backend syntax passes (`python -m py_compile backend.py`)
- [ ] Python tests pass (`pytest -v`)
- [ ] Frontend builds (`cd frontend && npm run build`)
- [ ] README/docs updated
- [ ] No local artifacts included

## Code Style

- Prefer small focused functions.
- Keep names descriptive and consistent.
- Preserve backward compatibility where possible.
- Add comments only where logic is non-obvious.
