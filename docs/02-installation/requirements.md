# System Requirements

**Pipeline Stage**: 2 - Installation

**Prerequisites**: 
- [Installation Overview](index.md)

This page details the system requirements for running Docling Graph.

## Python Requirements

### Supported Versions

- **Python 3.10** ✅
- **Python 3.11** ✅
- **Python 3.12** ✅

### Check Your Python Version

```bash
python --version
# or
python3 --version
```

### Installing Python

If you need to install or upgrade Python:

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

**macOS**:
```bash
brew install python@3.10
```

**Windows**:
Download from [python.org](https://www.python.org/downloads/)

## Hardware Requirements

### Minimum Configuration

For basic usage (VLM with small documents or remote LLM):

| Component | Requirement |
|-----------|-------------|
| **CPU** | 4 cores, 2.0 GHz+ |
| **RAM** | 8 GB |
| **Disk** | 5 GB free space |
| **GPU** | Not required (for remote LLM only) |
| **Network** | Required for remote LLM |

### Recommended Configuration

For optimal performance with local inference:

| Component | Requirement |
|-----------|-------------|
| **CPU** | 8+ cores, 3.0 GHz+ |
| **RAM** | 16 GB or more |
| **Disk** | 20 GB free space (for models) |
| **GPU** | NVIDIA GPU with 8+ GB VRAM |
| **CUDA** | 11.8 or 12.1 |
| **Network** | Optional (for remote LLM) |

### GPU Requirements by Use Case

#### VLM Only (NuExtract)

| Model | VRAM Required | Recommended GPU |
|-------|---------------|-----------------|
| NuExtract-2B | 4 GB | GTX 1650, RTX 3050 |
| NuExtract-8B | 8 GB | RTX 3060, RTX 4060 |

#### Local LLM (vLLM)

| Model Size | VRAM Required | Recommended GPU |
|------------|---------------|-----------------|
| 1B-4B params | 8 GB | RTX 3060, RTX 4060 |
| 7B-8B params | 16 GB | RTX 3080, RTX 4070 Ti |
| 13B+ params | 24 GB+ | RTX 3090, RTX 4090, A100 |

#### Remote LLM Only

| Requirement | Value |
|-------------|-------|
| GPU | Not required |
| Network | Stable internet connection |
| API Keys | Required for chosen provider |

## Operating System Requirements

### Linux

**Supported Distributions**:
- Ubuntu 20.04, 22.04, 24.04
- Debian 11, 12
- CentOS 8+
- Fedora 35+
- Arch Linux (latest)

**Required Packages**:
```bash
# Ubuntu/Debian
sudo apt install build-essential python3-dev git

# CentOS/Fedora
sudo dnf install gcc gcc-c++ python3-devel git

# Arch
sudo pacman -S base-devel python git
```

### macOS

**Supported Versions**:
- macOS 11 (Big Sur) or later
- macOS 12 (Monterey)
- macOS 13 (Ventura)
- macOS 14 (Sonoma)

**Required Tools**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Note**: GPU acceleration not available on macOS (Apple Silicon or Intel). Use remote LLM for best performance.

### Windows

**Supported Versions**:
- Windows 10 (version 1903 or later)
- Windows 11

**Recommended Setup**:
- **WSL2** (Windows Subsystem for Linux) for best compatibility
- **Native Windows** supported but WSL2 recommended

**WSL2 Setup**:
```powershell
# Enable WSL2
wsl --install

# Install Ubuntu
wsl --install -d Ubuntu-22.04

# Inside WSL2, follow Linux instructions
```

**Native Windows Requirements**:
- Visual Studio Build Tools or Visual Studio 2019+
- Git for Windows
- CUDA Toolkit (for GPU support)

## GPU and CUDA Requirements

### NVIDIA GPU

**Supported GPUs**:
- GeForce RTX 20/30/40 series
- GeForce GTX 16 series (limited)
- Quadro RTX series
- Tesla/A100/H100 series

**Check GPU**:
```bash
# Linux
nvidia-smi

# Windows
nvidia-smi.exe
```

### CUDA Toolkit

**Supported Versions**:
- CUDA 11.8 (recommended)
- CUDA 12.1 (recommended)
- CUDA 12.2+

**Check CUDA Version**:
```bash
nvcc --version
# or
nvidia-smi
```

**Installation**: See [GPU Setup Guide](gpu-setup.md)

### AMD GPU

**Status**: Not currently supported
- AMD ROCm support planned for future release
- Use remote LLM as alternative

### Apple Silicon (M1/M2/M3)

**Status**: Limited support
- VLM works via CPU (slower)
- Local LLM not optimized
- **Recommended**: Use remote LLM providers

## Network Requirements

### For Remote LLM

| Requirement | Specification |
|-------------|---------------|
| **Bandwidth** | 1 Mbps minimum, 10+ Mbps recommended |
| **Latency** | < 200ms to provider |
| **Stability** | Consistent connection required |
| **Firewall** | Allow HTTPS (port 443) |

### For Local Inference

| Requirement | Specification |
|-------------|---------------|
| **Network** | Optional (for downloading models) |
| **Bandwidth** | Only needed for initial model download |

## Disk Space Requirements

### Base Installation

```
Core Package:           ~500 MB
Python Dependencies:    ~2 GB
Total:                  ~2.5 GB
```

### With Models

```
VLM Models:
  - NuExtract-2B:       ~4 GB
  - NuExtract-8B:       ~16 GB

LLM Models (examples):
  - Granite-1B:         ~2 GB
  - Llama-7B:           ~14 GB
  - Llama-13B:          ~26 GB

Recommended Free Space: 20 GB
```

### For Processing

```
Temporary Files:        ~1 GB per document
Output Files:           ~100 MB per document
Cache:                  ~500 MB

Recommended Free Space: 5 GB for active processing
```

## Memory (RAM) Requirements

### By Use Case

| Use Case | Minimum RAM | Recommended RAM |
|----------|-------------|-----------------|
| Remote LLM only | 4 GB | 8 GB |
| VLM (NuExtract-2B) | 8 GB | 16 GB |
| VLM (NuExtract-8B) | 12 GB | 24 GB |
| Local LLM (1B-4B) | 16 GB | 32 GB |
| Local LLM (7B-8B) | 24 GB | 48 GB |
| Local LLM (13B+) | 32 GB | 64 GB+ |

### Memory Usage Patterns

```
Base Process:           ~500 MB
Document Processing:    ~1-2 GB per document
Model Loading:          Varies by model size
Graph Construction:     ~100 MB per 1000 nodes
```

## Software Dependencies

### Required

- **Python**: 3.10, 3.11, or 3.12
- **uv**: Package manager (installed automatically)
- **Git**: For cloning repository

### Optional (Installed by uv)

- **PyTorch**: For GPU acceleration
- **CUDA Toolkit**: For NVIDIA GPU support
- **Docling**: Document conversion
- **NetworkX**: Graph operations
- **Pydantic**: Data validation

## Compatibility Matrix

### Backend Compatibility

| Backend | Linux | macOS | Windows | GPU Required |
|---------|-------|-------|---------|--------------|
| VLM (NuExtract) | ✅ | ✅ | ✅ | Recommended |
| LLM (vLLM) | ✅ | ❌ | ⚠️ | Yes |
| LLM (Ollama) | ✅ | ✅ | ✅ | Optional |
| LLM (Remote APIs) | ✅ | ✅ | ✅ | No |

Legend:
- ✅ Fully supported
- ⚠️ Supported with limitations
- ❌ Not supported

### Provider Compatibility

| Provider | Local | Remote | GPU Required | API Key Required |
|----------|-------|--------|--------------|------------------|
| NuExtract (VLM) | ✅ | ❌ | Recommended | No |
| vLLM | ✅ | ❌ | Yes | No |
| Ollama | ✅ | ❌ | Optional | No |
| OpenAI | ❌ | ✅ | No | Yes |
| Mistral | ❌ | ✅ | No | Yes |
| Gemini | ❌ | ✅ | No | Yes |
| WatsonX | ❌ | ✅ | No | Yes |

## Verification Checklist

Before proceeding with installation, verify:

- [ ] Python 3.10+ installed
- [ ] Sufficient disk space (5 GB minimum, 20 GB recommended)
- [ ] Sufficient RAM (8 GB minimum, 16 GB recommended)
- [ ] GPU available (if using local inference)
- [ ] CUDA installed (if using NVIDIA GPU)
- [ ] Network connection (if using remote LLM)
- [ ] API keys ready (if using remote LLM)

## Performance Expectations

### Document Processing Speed

| Configuration | Small Doc (1-2 pages) | Large Doc (10+ pages) |
|---------------|----------------------|----------------------|
| VLM (GPU) | 5-10 seconds | 30-60 seconds |
| VLM (CPU) | 30-60 seconds | 3-5 minutes |
| LLM Local (GPU) | 10-30 seconds | 1-3 minutes |
| LLM Remote | 5-15 seconds | 30-90 seconds |

### Memory Usage

| Configuration | Peak Memory | Sustained Memory |
|---------------|-------------|------------------|
| Remote LLM | 2-4 GB | 1-2 GB |
| VLM (2B) | 6-8 GB | 4-6 GB |
| VLM (8B) | 12-16 GB | 10-12 GB |
| Local LLM (7B) | 20-24 GB | 16-20 GB |

## Troubleshooting

### Check System Resources

```bash
# Check RAM
free -h  # Linux
vm_stat  # macOS

# Check disk space
df -h

# Check GPU
nvidia-smi  # NVIDIA
```

### Insufficient Resources

**Problem**: Not enough RAM/VRAM

**Solutions**:
1. Use remote LLM instead of local
2. Use smaller models (NuExtract-2B instead of 8B)
3. Enable chunking to reduce memory usage
4. Close other applications

**Problem**: No GPU available

**Solutions**:
1. Use remote LLM providers (no GPU needed)
2. Use Ollama with CPU (slower but works)
3. Consider cloud GPU instances

## Next Steps

Requirements verified? Continue with:

1. **[Basic Setup](basic-setup.md)** - Install Docling Graph
2. **[GPU Setup](gpu-setup.md)** - Configure CUDA (if using GPU)
3. **[API Keys](api-keys.md)** - Set up remote providers (if using APIs)

## Related Documentation

- **[Installation Overview](index.md)**: Installation options
- **[GPU Setup](gpu-setup.md)**: Detailed GPU configuration
- **[Troubleshooting](index.md#common-issues)**: Common installation issues

---

**Questions about requirements?** Check the [installation overview](index.md) or proceed to [basic setup](basic-setup.md)!