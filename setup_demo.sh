#!/bin/bash
# Demo Setup Script for Panel Presentation
# Run this 5 minutes before your presentation

echo "🎯 Setting up Project Cerberus Demo..."
echo ""

# Navigate to project directory
cd /Users/admin/Desktop/major_projekt || exit 1

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Verify installation
echo ""
echo "✅ Verifying Phase 2 installation..."
python3 scripts/test_phase2.py

# Check if figures exist
echo ""
echo "📊 Checking generated figures..."
if [ -d "figures" ] && [ "$(ls -A figures/*.png 2>/dev/null)" ]; then
    echo "✅ Figures found:"
    ls -1 figures/*.png
else
    echo "⚠️  No figures found. Generate them with:"
    echo "   docker run --rm -v \"\$(pwd)/figures:/app/figures\" cerberus-figures python scripts/generate_figures.py --device cpu --fgsm-eps 0.03 --max-samples 100"
fi

# Check Docker
echo ""
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker installed"
    if docker images | grep -q cerberus; then
        echo "✅ Cerberus Docker image found"
    else
        echo "⚠️  Cerberus Docker image not found. Build with:"
        echo "   docker build -t cerberus-figures -f Dockerfile.figures ."
    fi
else
    echo "⚠️  Docker not installed or not running"
fi

# Check key files
echo ""
echo "📁 Checking key files..."
files=(
    "cerberus/adversarial_training.py"
    "cerberus/baseline_training.py"
    "configs/training_config.yaml"
    "scripts/compare_models.py"
    "scripts/plot_training_curves.py"
    "README.md"
    "PHASE2_IMPLEMENTATION.md"
)

all_good=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file - MISSING!"
        all_good=false
    fi
done

# Show git status
echo ""
echo "🔍 Git status:"
git log --oneline -3

# Summary
echo ""
echo "========================================="
if $all_good; then
    echo "✅ ALL SYSTEMS READY FOR DEMO!"
else
    echo "⚠️  SOME FILES MISSING - CHECK ABOVE"
fi
echo "========================================="
echo ""
echo "💡 Quick commands:"
echo "   cat configs/training_config.yaml"
echo "   head -50 cerberus/adversarial_training.py"
echo "   ls -lh figures/*.png"
echo "   open figures/"
echo ""
echo "📖 Full guide: DEMO_GUIDE.md"
echo "📋 Cheat sheet: DEMO_CHEAT_SHEET.md"
echo ""
echo "Good luck with your presentation! 🎓🚀"
