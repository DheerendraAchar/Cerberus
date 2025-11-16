FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./

# Install CPU-only PyTorch wheels from the official PyTorch CPU index to avoid
# large CUDA/CuDNN downloads in CPU-only environments.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
	pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
		"torch==2.9.1+cpu" "torchvision==0.24.1+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
	pip install --no-cache-dir -r requirements.txt

COPY . /app
CMD ["python", "run_demo.py", "--config", "configs/sample_config.yaml"]
