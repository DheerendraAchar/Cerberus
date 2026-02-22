# 🎯 QUICK DEMO CHEAT SHEET
## Project Cerberus - Panel Presentation

**Last Minute Setup: 5 minutes before panel**

---

## 1️⃣ OPEN THESE FIRST

```bash
cd /Users/admin/Desktop/major_projekt
source .venv/bin/activate

# Open VS Code with README
code README.md

# Open figures folder (keep minimized)
open figures/

# Open terminal in fullscreen
# Increase terminal font size: Cmd+Plus or Ctrl+Plus
```

---

## 2️⃣ VERIFY EVERYTHING WORKS

```bash
# This should take 10 seconds
python3 scripts/test_phase2.py
```

**Expected: All ✅ green checkmarks**

---

## 3️⃣ DEMONSTRATION SEQUENCE

### **Demo 1: Show Project (30 seconds)**
```bash
# Show files
ls -la

# Show structure
tree -L 2 -I '__pycache__|*.pyc|.venv|.git' .
# OR if tree not installed:
ls cerberus/
ls configs/
ls scripts/
```

### **Demo 2: Show Configuration (30 seconds)**
```bash
cat configs/training_config.yaml
```
**Point out: epsilon=0.03, alpha=0.5**

### **Demo 3: Show Training Code (1 minute)**
```bash
head -50 cerberus/adversarial_training.py
```
**Point out: Lines 30-40 show custom FGSM implementation**

### **Demo 4: Show Visualizations (2 minutes)**
```bash
ls -lh figures/*.png
open figures/per_class_robustness_eps0.03.png
open figures/fgsm_examples_eps0.03.png
open figures/attack_success_vs_epsilon.png
```

### **Demo 5: Quick Training (OPTIONAL - 5 minutes)**
```bash
# Only if panel wants to see live training
python3 run_demo.py \
    --mode train \
    --training-type adversarial \
    --config configs/training_config.yaml \
    --num-epochs 2
```

**OR just show the command:**
```bash
echo "Full training command (takes 3-4 hours):"
echo "python3 run_demo.py --mode train --training-type adversarial --config configs/training_config.yaml"
```

### **Demo 6: Docker (1 minute)**
```bash
docker images | grep cerberus
```

---

## 4️⃣ KEY TALKING POINTS

**When showing code:**
> "This is custom implementation - we wrote 2,200+ lines, not just library wrappers"

**When showing results:**
> "We achieved 18% robustness improvement: from 8.5% to 58.8% adversarial accuracy"

**When asked about innovation:**
> "65% original implementation - complete training pipeline from scratch, not just evaluation"

**When asked about timeline:**
> "60% complete, ahead of schedule - Phase 2 done in 1 day instead of planned 6 days"

---

## 5️⃣ RESULTS TO MEMORIZE

| Metric | Baseline | Adversarial | Improvement |
|--------|----------|-------------|-------------|
| Clean Acc | 92.5% | 88.2% | -4.3% |
| Robust Acc | 8.5% | **58.8%** | **+50.3%** ⭐ |

**Code Stats:**
- 2,200+ lines custom code
- 3,000+ lines documentation
- 8 unit tests (100% pass)
- 60% project complete

---

## 6️⃣ IF SOMETHING BREAKS

**Can't run training?**
→ Show figures that are already generated

**Docker not working?**
→ Skip Docker demo, focus on Python code

**Code not displaying well?**
→ Open in VS Code and zoom in

**Terminal freezes?**
→ Open new terminal, cd to project, activate venv

---

## 7️⃣ Q&A QUICK ANSWERS

**Q: Why CIFAR-10 only?**
**A:** "Standard benchmark, computationally manageable, allows reproducible comparison with research. ImageNet needs 50+ hours on GPU."

**Q: How is this different from IBM ART?**
**A:** "ART provides attacks only. We implemented complete training pipeline - 350 lines of adversarial training code."

**Q: Computation cost?**
**A:** "3-4 hours on CPU, 45 minutes on GPU. 4GB RAM. Cost-effective for academic use."

**Q: Limitations?**
**A:** "4% clean accuracy trade-off, longer training time, not provable guarantees. But most practical defense available."

**Q: Publication plans?**
**A:** "IEEE SSCI or ICMLA by June 2026. Need Phase 3 transfer analysis. Currently workshop level."

---

## 8️⃣ EMERGENCY BACKUP

**If live demo completely fails:**

1. Show figures folder (already open)
2. Show README in VS Code
3. Walk through PRESENTATION_CONTENT.md
4. Show printed screenshots (if you prepared them)
5. Explain what WOULD happen if running

---

## 9️⃣ CLOSING STATEMENT

> "To summarize: We've built a complete ML pipeline with 2,200+ lines of custom code achieving 18% robustness improvement. We have production-quality implementation with Docker, CI/CD, comprehensive testing, and extensive documentation. We're 60% complete and ahead of schedule. Thank you!"

---

## 🎯 FINAL CHECKLIST

Before walking into the room:

- [ ] Laptop charged or plugged in
- [ ] Terminal open in project folder
- [ ] Virtual env activated
- [ ] test_phase2.py passed
- [ ] Figures folder open (minimized)
- [ ] VS Code with README open
- [ ] Notifications disabled
- [ ] Terminal font size increased
- [ ] Water bottle (stay hydrated!)
- [ ] Deep breath - you've got this! 💪

---

**Print this page and keep it with you!**

Good luck! 🎓🚀
