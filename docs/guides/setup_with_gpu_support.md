# Installing torch with GPU Support

## Important Notice: Package Conflict with uv

Currently, there is a **package conflict** preventing the use of PyTorch with GPU support via `uv` together with `docling[vlm]`. Installing GPU-enabled torch through `uv` sync alongside `docling[vlm]` causes dependency resolution failures.


## Manual Workaround for GPU Support

If you need GPU support, follow these manual steps:

1. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   # On Windows PowerShell
   .\venv\Scripts\Activate
   # On Linux/macOS
   source venv/bin/activate
   ```

2. **Install your project in editable mode:**

   Navigate to the root directory of your project (where `setup.py` or `pyproject.toml` is), then run:

   ```bash
   pip install -e .
   ```

3. **Uninstall CPU-only PyTorch packages:**

   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```

4. **Check your CUDA version:**

   Use NVIDIA’s system management tool:

   ```bash
   nvidia-smi
   ```

   Look for something like:

   ```
   CUDA Version: 13.0
   ```

5. **Install GPU-enabled PyTorch packages:**

   Visit the official [PyTorch installation page](https://pytorch.org/get-started/locally/) for commands matching your CUDA version.

   Example for CUDA 13.0:

   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   ```


## Verify Installation

Run this command to verify that PyTorch is installed correctly with CUDA support:

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Example output:

```
PyTorch version: 2.9.0+cu130
CUDA available: True
```


## Important CLI Usage Change

Moving forward, **do not use `uv run docling-graph <command>`**. Instead, directly call the docling-graph CLI with:

```bash
docling-graph <command> [OPTIONS]
```

This ensures commands run correctly without the `uv` wrapper in GPU-enabled environments.


## CLI Usage Examples (Direct Calls)

### 1. Initialize Configuration

```bash
docling-graph init
```

### 2. Run Conversion

```bash
docling-graph convert <SOURCE_FILE_PATH> --template "<TEMPLATE_PATH>" [OPTIONS]
```

### 3. Inspect Output

```bash
docling-graph inspect <CONVERT_OUTPUT_PATH>
```