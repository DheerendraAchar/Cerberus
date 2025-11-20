#!/bin/bash
# Generate figures using Docker container (includes all dependencies)

set -e

echo "Building Docker image with figure generation script..."
docker build -t cerberus-figures -f Dockerfile.figures .

echo ""
echo "Running figure generation in container..."
docker run --rm -v "$(pwd)/figures:/app/figures" cerberus-figures

echo ""
echo "Figures generated successfully in ./figures/"
ls -lh figures/
