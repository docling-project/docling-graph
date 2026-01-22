# GPU Setup

**Pipeline Stage**: 2 - Installation

**Prerequisites**: 
- [Installation Overview](index.md)
- [System Requirements](requirements.md)
- [Basic Setup](basic-setup.md)

This page guides you through setting up GPU support for local inference with Docling Graph.

## Overview

GPU acceleration significantly improves performance for:
- **VLM Backend**: NuExtract models (4-8 GB VRAM)
- **Local LLM**: vLLM inference (8-24 GB VRAM)

**Note**: Remote LLM providers (OpenAI, Mistral, Gemini, WatsonX) do not require GPU.

## Important: Package Conflict Notice

⚠️ **Current Limitation**: There is a package conflict preventing the use of PyTorch with GPU support via `uv` together with `docling[vlm]`. Installing GPU-enabled torch through `uv sync` alongside `docling[vlm]` causes dependency resolution failures.

**Workaround**: Manual installation using `pip` (see below).

## Prerequisites

### 1. NVIDIA GPU

Verify you have a compatible NVIDIA GPU:

```bash
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    15W / 250W |    500MiB /  8192MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
```

### 2. CUDA Toolkit

Check your CUDA version from `nvidia-smi` output above.

**Supported CUDA Versions**:
- CUDA 11.8 (recommended)
- CUDA 12.1 (recommended)
- CUDA 12.2+

### 3. CUDA Toolkit Installation

If CUDA is not installed:

**Linux (Ubuntu/Debian)**:
```bash
# CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Windows**:
1. Download CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Run installer
3. Verify installation: `nvcc --version`

## Manual GPU Setup (Workaround)

Due to the package conflict, follow these steps for GPU support:

### Step 1: Create Virtual Environment

```bash
# Navigate to docling-graph directory
cd docling-graph

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate

# Windows PowerShell:
.\.venv\Scripts\Activate

# Windows CMD:
.venv\Scripts\activate.bat
```

### Step 2: Install Docling Graph

Choose the installation that matches your needs:

**Minimal (VLM only)**:
```bash
pip install -e .
```

**Full (all features)**:
```bash
pip install -e .[all]
```

**Local LLM only**:
```bash
pip install -e .[local]
```

**Remote API only**:
```bash
pip install -e .[remote]
```

### Step 3: Uninstall CPU-only PyTorch

```bash
pip uninstall torch torchvision torchaudio -y
```

### Step 4: Install GPU-enabled PyTorch

Visit [PyTorch installation page](https://pytorch.org/get-started/locally/) for the exact command matching your CUDA version.

**CUDA 11.8**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.2+**:
```bash
pip3 install torch torchvision torchaudio
```

### Step 5: Verify GPU Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch version: 2.2.0+cu121
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: NVIDIA GeForce RTX 3060
```

## CLI Usage with GPU Setup

⚠️ **Important**: When using the manual GPU setup (pip-based virtual environment), **do not use `uv run`**. Instead, call commands directly:

### Correct Usage (with GPU setup)

```bash
# Activate virtual environment first
source .venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\Activate  # Windows

# Then use direct commands
docling-graph --version
docling-graph init
docling-graph convert document.pdf --template "templates.Invoice"
docling-graph inspect outputs
```

### Incorrect Usage (will not work)

```bash
# Don't use uv run with manual GPU setup
uv run docling-graph convert document.pdf  # ❌ Wrong
```

## Testing GPU Performance

### Test VLM with GPU

```bash
# Activate virtual environment
source .venv/bin/activate

# Run VLM example
python docs/examples/scripts/01_vlm_from_image.py
```

Monitor GPU usage:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Test Local LLM with GPU

```bash
# Activate virtual environment
source .venv/bin/activate

# Start vLLM server (if using vLLM)
python -m vllm.entrypoints.openai.api_server \
    --model ibm-granite/granite-4.0-1b \
    --port 8000

# In another terminal, run extraction
docling-graph convert document.pdf \
    --template "templates.Invoice" \
    --backend llm \
    --inference local \
    --provider vllm
```

## Performance Expectations

### VLM Performance

| Model | GPU | Processing Speed (per page) |
|-------|-----|----------------------------|
| NuExtract-2B | RTX 3060 (8GB) | 2-3 seconds |
| NuExtract-2B | RTX 4090 (24GB) | 1-2 seconds |
| NuExtract-8B | RTX 3060 (8GB) | 5-7 seconds |
| NuExtract-8B | RTX 4090 (24GB) | 2-3 seconds |

### Local LLM Performance

| Model Size | GPU | Processing Speed (per chunk) |
|------------|-----|------------------------------|
| 1B-4B | RTX 3060 (8GB) | 5-10 seconds |
| 1B-4B | RTX 4090 (24GB) | 2-5 seconds |
| 7B-8B | RTX 3080 (16GB) | 10-20 seconds |
| 7B-8B | RTX 4090 (24GB) | 5-10 seconds |

## Troubleshooting

### Issue: CUDA not available

**Check**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**If False**:

1. **Verify NVIDIA driver**:
   ```bash
   nvidia-smi
   ```

2. **Check CUDA installation**:
   ```bash
   nvcc --version
   ```

3. **Reinstall PyTorch with correct CUDA version**:
   ```bash
   pip uninstall torch torchvision torchaudio -y
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue: Out of memory

**Symptoms**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Use smaller model**:
   ```bash
   # Use NuExtract-2B instead of 8B
   # Or use smaller LLM
   ```

2. **Enable chunking**:
   ```bash
   docling-graph convert document.pdf \
       --template "templates.Invoice" \
       --use-chunking
   ```

3. **Reduce batch size**:
   ```python
   config = PipelineConfig(
       max_batch_size=1,  # Process one chunk at a time
       use_chunking=True
   )
   ```

4. **Clear GPU memory**:
   ```bash
   # Kill other GPU processes
   nvidia-smi
   # Note PIDs and kill if needed
   kill -9 <PID>
   ```

### Issue: Slow performance

**Check GPU utilization**:
```bash
nvidia-smi
```

**If GPU utilization is low**:

1. **Verify GPU is being used**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.current_device())
   ```

2. **Check for CPU fallback**:
   - Look for warnings in output
   - Verify PyTorch CUDA version matches system CUDA

3. **Optimize batch size**:
   - Increase batch size if memory allows
   - Monitor with `nvidia-smi`

### Issue: Driver version mismatch

**Symptoms**:
```
CUDA driver version is insufficient for CUDA runtime version
```

**Solution**:
```bash
# Update NVIDIA driver
# Ubuntu/Debian:
sudo apt update
sudo apt install nvidia-driver-535

# Or download from NVIDIA website
```

## GPU Memory Management

### Monitor Memory Usage

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use Python
python -c "import torch; print(f'Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB'); print(f'Reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB')"
```

### Clear GPU Cache

```python
import torch
torch.cuda.empty_cache()
```

### Best Practices

1. **Process documents sequentially** for large batches
2. **Use chunking** for large documents
3. **Monitor memory** with `nvidia-smi`
4. **Close unused processes** to free VRAM
5. **Use appropriate model size** for your GPU

## Alternative: Cloud GPU

If you don't have a local GPU, consider cloud options:

### Google Colab

```python
# In Colab notebook
!git clone https://github.com/IBM/docling-graph
%cd docling-graph
!pip install -e .[all]

# GPU is automatically available
import torch
print(torch.cuda.is_available())  # Should be True
```

### AWS/Azure/GCP

- Launch GPU instance (e.g., g4dn.xlarge on AWS)
- Follow Linux installation instructions
- GPU drivers usually pre-installed

## Next Steps

GPU setup complete! Now:

1. **[API Keys](api-keys.md)** (optional) - Set up remote providers
2. **[Schema Definition](../03-schema-definition/index.md)** - Create your first template
3. **[Quick Start](../09-examples/quickstart.md)** - Run your first extraction

## Related Documentation

- **[Requirements](requirements.md)**: GPU requirements
- **[Basic Setup](basic-setup.md)**: Installation without GPU
- **[API Keys](api-keys.md)**: Remote provider setup (no GPU needed)
- **[Troubleshooting](index.md#common-issues)**: Common issues

---

**GPU working?** Continue to [schema definition](../03-schema-definition/index.md) or try the [quickstart](../09-examples/quickstart.md)!