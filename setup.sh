#!/bin/bash

# Cerberus Full-Stack Application Quick Start Script
# This script sets up and starts both the backend and frontend

set -e

echo "🚀 Cerberus Full-Stack Application - Quick Start"
echo "================================================"
echo ""

# Check Python
echo "✓ Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $PYTHON_VERSION"

# Check Node.js
echo "✓ Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 14 or higher."
    exit 1
fi
NODE_VERSION=$(node --version)
echo "  Found Node.js $NODE_VERSION"

# Check npm
echo "✓ Checking npm..."
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed."
    exit 1
fi
NPM_VERSION=$(npm --version)
echo "  Found npm $NPM_VERSION"

echo ""
echo "📦 Installing Backend Dependencies..."
pip install -q flask flask-cors torch torchvision matplotlib numpy pillow 2>/dev/null || {
    echo "⚠️  Some dependencies may have failed to install. Please run manually:"
    echo "   pip install flask flask-cors torch torchvision matplotlib numpy pillow"
}

echo "✓ Backend dependencies installed"

echo ""
echo "📦 Installing Frontend Dependencies..."
cd frontend
npm install --quiet > /dev/null 2>&1 || npm install
echo "✓ Frontend dependencies installed"

echo ""
echo "================================================"
echo "✅ Setup Complete!"
echo "================================================"
echo ""
echo "🎯 Next Steps:"
echo ""
echo "1. Start Backend (Terminal 1):"
echo "   python backend.py"
echo ""
echo "2. Start Frontend (Terminal 2):"
echo "   cd frontend && npm start"
echo ""
echo "3. Open Browser:"
echo "   http://localhost:3000"
echo ""
echo "📖 For detailed instructions, see STARTUP_GUIDE.md"
echo ""
