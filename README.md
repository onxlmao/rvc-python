# RVC-Python

A high-performance Python library for **RVC (Retrieval-based Voice Conversion)** — convert voices via CLI, Python API, or a REST API server.

Supports **Python 3.12+**, GPU (CUDA/MPS/XPU) acceleration, and multiple pitch extraction algorithms.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
  - [CPU-Only Installation](#cpu-only-installation)
  - [GPU Installation (CUDA)](#gpu-installation-cuda)
  - [GPU Installation (ROCm / AMD)](#gpu-installation-rocm--amd)
  - [Intel XPU (IPEX)](#intel-xpu-ipex)
  - [Apple Silicon (MPS)](#apple-silicon-mps)
  - [From Source](#from-source)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Python Module](#python-module)
  - [REST API Server](#rest-api-server)
- [API Reference](#api-reference)
  - [Endpoints](#endpoints)
  - [Python Client Examples](#python-client-examples)
- [Python SDK Reference](#python-sdk-reference)
  - [RVCInference Class](#rvcinference-class)
  - [Legacy Functions](#legacy-functions)
- [Model Management](#model-management)
  - [Directory Structure](#directory-structure)
  - [Model Formats](#model-formats)
  - [Model Versions](#model-versions)
- [Pitch Extraction Methods](#pitch-extraction-methods)
- [Configuration & Parameters](#configuration--parameters)
  - [CLI Parameters](#cli-parameters)
  - [Python Parameters](#python-parameters)
- [Advanced Topics](#advanced-topics)
  - [Performance Optimization](#performance-optimization)
  - [Batch Processing](#batch-processing)
  - [ONNX Export](#onnx-export)
  - [TorchScript JIT Compilation](#torchscript-jit-compilation)
  - [DirectML Support (Windows)](#directml-support-windows)
  - [IPEX Support (Intel GPUs)](#ipex-support-intel-gpus)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

RVC-Python provides a clean interface to the RVC voice conversion engine. It loads pre-trained RVC models (v1 and v2) and performs real-time or batch voice conversion on audio files.

The library wraps the core RVC inference pipeline — HuBERT feature extraction, VITS-based synthesis, pitch detection, and FAISS-based index retrieval — into an easy-to-use Python class and a FastAPI-based REST server.

### What is RVC?

Retrieval-based Voice Conversion (RVC) is a voice cloning and conversion technique that uses a pre-trained HuBERT model for speaker feature extraction combined with a VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) synthesizer. It supports both singing voice conversion and speech-to-speech conversion with high fidelity.

---

## Features

- **Multiple inference modes**: CLI, Python SDK, and REST API
- **Python 3.12+ support**: Full compatibility with the latest Python releases
- **GPU acceleration**: CUDA (NVIDIA), MPS (Apple Silicon), XPU (Intel via IPEX), DirectML (Windows)
- **Four pitch extraction algorithms**: RMVPE (recommended), Harvest, Crepe, PM (Parselmouth)
- **Dynamic model management**: Load, unload, and switch models at runtime
- **Batch processing**: Convert entire directories of audio files
- **V1 and V2 model support**: Full compatibility with both RVC model generations
- **Index-based retrieval**: Optional FAISS index for improved voice similarity
- **ONNX export**: Export models for deployment outside of PyTorch
- **TorchScript JIT**: Compile models for faster inference
- **Configurable pipeline**: Fine-tune every aspect of the voice conversion process

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.12 | 3.12 or 3.13 |
| PyTorch | 2.3.1 | 2.4+ with CUDA 12.x |
| CUDA Toolkit | — | 12.1+ (for GPU) |
| FFmpeg | 5.x+ | 7.x+ |
| RAM | 4 GB | 8 GB+ |
| GPU VRAM | — | 4 GB+ (for half-precision) |

### Supported Platforms

- **Linux** (x86_64, aarch64)
- **Windows** (x86_64)
- **macOS** (Apple Silicon M1+, Intel)

---

## Installation

### CPU-Only Installation

```bash
pip install git+https://github.com/onxlmao/rvc-python.git
```

### GPU Installation (CUDA)

For NVIDIA GPUs with CUDA 12.x:

```bash
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Then install rvc-python
pip install git+https://github.com/onxlmao/rvc-python.git
```

For NVIDIA GPUs with CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/onxlmao/rvc-python.git
```

### GPU Installation (ROCm / AMD)

For AMD GPUs on Linux:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install git+https://github.com/onxlmao/rvc-python.git
```

### Intel XPU (IPEX)

For Intel Arc GPUs and data center GPUs:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
pip install git+https://github.com/onxlmao/rvc-python.git
pip install intel_extension_for_pytorch
```

### Apple Silicon (MPS)

MPS is automatically detected and used on Apple Silicon Macs:

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/onxlmao/rvc-python.git
```

### From Source

```bash
git clone https://github.com/onxlmao/rvc-python.git
cd rvc-python
pip install -e .
```

> **Note**: The `fairseq` dependency is installed from a fork (`fumiama/fairseq`) that includes fixes for RVC compatibility. This is automatically handled by pip.

---

## Quick Start

### CLI — Convert a single file

```bash
python -m rvc_python cli \
  -i input.wav \
  -o output.wav \
  -mp model.pth \
  -de cuda:0 \
  -me rmvpe \
  -pi 2
```

### Python SDK — Basic usage

```python
from rvc_python.infer import RVCInference

# Initialize with GPU
rvc = RVCInference(device="cuda:0")

# Load a model
rvc.load_model("path/to/model.pth", version="v2")

# Adjust parameters
rvc.set_params(f0up_key=2, f0method="rmvpe", protect=0.5)

# Convert a single file
rvc.infer_file("input.wav", "output.wav")

# Convert a directory
rvc.infer_dir("input_folder/", "output_folder/")

# Unload to free memory
rvc.unload_model()
```

### REST API — Start the server

```bash
python -m rvc_python api -p 5050 -l -pm model_name
```

```python
import requests, base64

# Convert audio via API
with open("input.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

resp = requests.post("http://localhost:5050/convert", json={"audio_data": audio_b64})
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

---

## Usage

### Command Line Interface (CLI)

The CLI provides two subcommands: `cli` for direct file processing and `api` for starting a REST server.

```bash
python -m rvc_python {cli,api} [options]
```

#### Process a single file

```bash
python -m rvc_python cli \
  --input input.wav \
  --output output.wav \
  --model path/to/model.pth \
  --index path/to/model.index \
  --device cuda:0 \
  --method rmvpe \
  --pitch 0 \
  --index_rate 0.6 \
  --filter_radius 3 \
  --resample_sr 0 \
  --rms_mix_rate 0.25 \
  --protect 0.5 \
  --version v2
```

#### Process a directory

```bash
python -m rvc_python cli \
  --dir input_audio/ \
  --output output_audio/ \
  --model path/to/model.pth \
  --device cuda:0
```

#### Start the API server

```bash
# Listen on localhost only
python -m rvc_python api --port 5050 --models_dir ./rvc_models

# Listen on all interfaces (accessible from network)
python -m rvc_python api --port 5050 --listen --models_dir ./rvc_models

# Preload a model on startup
python -m rvc_python api --port 5050 --preload-model my_model
```

### Python Module

#### Basic inference

```python
from rvc_python.infer import RVCInference

rvc = RVCInference(
    models_dir="rvc_models",   # Directory containing model subdirectories
    device="cuda:0",           # "cpu:0", "cuda:0", "mps", "xpu:0"
    model_path=None,           # Load a model on init (optional)
    index_path="",             # Path to .index file (optional)
    version="v2"               # Model version: "v1" or "v2"
)

# List available models
print(rvc.list_models())

# Load a model by name (from models_dir)
rvc.load_model("my_voice_model")

# Or load by direct path
rvc.load_model("/path/to/model.pth", version="v2", index_path="/path/to/index.index")

# Set inference parameters
rvc.set_params(
    f0method="rmvpe",        # Pitch extraction: "rmvpe", "harvest", "crepe", "pm"
    f0up_key=0,              # Pitch shift in semitones (0 = no shift)
    index_rate=0.5,          # Feature search ratio (0.0 - 1.0)
    filter_radius=3,         # Median filter for pitch smoothing
    resample_sr=0,           # Output sample rate (0 = keep original)
    rms_mix_rate=1.0,        # Volume envelope mixing (0.0 - 1.0)
    protect=0.33             # Voiceless consonant protection (0.0 - 0.5)
)

# Convert a single file
rvc.infer_file("input.wav", "output.wav")

# Convert all files in a directory
output_files = rvc.infer_dir("input_dir/", "output_dir/")

# Switch to a different device
rvc.set_device("cpu:0")

# Unload model to free GPU memory
rvc.unload_model()
```

#### Changing models directory

```python
rvc.set_models_dir("/new/path/to/models")
print(rvc.list_models())
```

### REST API Server

The API server is built with FastAPI and provides full control over the RVC inference pipeline via HTTP endpoints.

#### Starting the server

```bash
python -m rvc_python api \
  --port 5050 \
  --listen \
  --models_dir ./rvc_models \
  --device cuda:0 \
  --preload-model my_model
```

The server includes CORS middleware and accepts JSON / multipart requests.

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/convert` | Convert base64-encoded audio |
| `POST` | `/convert_file` | Convert an uploaded audio file |
| `GET` | `/models` | List available models |
| `POST` | `/models/{model_name}` | Load a model by name |
| `GET` | `/params` | Get current parameters |
| `POST` | `/params` | Update parameters |
| `POST` | `/upload_model` | Upload a model ZIP file |
| `POST` | `/set_device` | Change computation device |
| `POST` | `/set_models_dir` | Change models directory |

### Python Client Examples

#### Convert audio (base64)

```python
import requests
import base64

# Read and encode audio
with open("input.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Send conversion request
resp = requests.post(
    "http://localhost:5050/convert",
    json={"audio_data": audio_b64}
)

# Save result
if resp.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(resp.content)
else:
    print(f"Error: {resp.json()['detail']}")
```

#### Convert audio (file upload)

```python
with open("input.wav", "rb") as f:
    resp = requests.post("http://localhost:5050/convert_file", files={"file": f})

with open("output.wav", "wb") as f:
    f.write(resp.content)
```

#### List and load models

```python
# List available models
models = requests.get("http://localhost:5050/models").json()
print(models)  # {"models": ["model_a", "model_b"]}

# Load a model
resp = requests.post("http://localhost:5050/models/model_a")
print(resp.json())  # {"message": "Model model_a loaded successfully"}
```

#### Get and set parameters

```python
# Get current parameters
params = requests.get("http://localhost:5050/params").json()
print(params)
# {"f0method": "rmvpe", "f0up_key": 0, "index_rate": 0.6, ...}

# Set parameters
requests.post("http://localhost:5050/params", json={
    "params": {
        "f0up_key": 3,
        "protect": 0.5,
        "f0method": "crepe"
    }
})
```

#### Upload a model

```python
with open("my_model.zip", "rb") as f:
    resp = requests.post(
        "http://localhost:5050/upload_model",
        files={"file": ("my_model.zip", f)}
    )
print(resp.json())
```

#### Change device

```python
requests.post("http://localhost:5050/set_device", json={"device": "cuda:0"})
```

#### Change models directory

```python
requests.post("http://localhost:5050/set_models_dir", json={"models_dir": "/path/to/models"})
```

---

## Python SDK Reference

### RVCInference Class

The main interface for RVC inference, located in `rvc_python.infer`.

#### Constructor

```python
RVCInference(
    models_dir: str = "rvc_models",
    device: str = "cpu:0",
    model_path: str | None = None,
    index_path: str = "",
    version: str = "v2"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models_dir` | `str` | `"rvc_models"` | Path to directory containing model subdirectories |
| `device` | `str` | `"cpu:0"` | Device for computation (`"cpu:0"`, `"cuda:0"`, `"mps"`, `"xpu:0"`) |
| `model_path` | `str \| None` | `None` | Path to a `.pth` model file to load on initialization |
| `index_path` | `str` | `""` | Path to an `.index` file for the model |
| `version` | `str` | `"v2"` | Model version (`"v1"` or `"v2"`) |

#### Methods

| Method | Description |
|--------|-------------|
| `load_model(path_or_name, version="v2", index_path="")` | Load an RVC model by name or file path |
| `unload_model()` | Unload the current model and free GPU memory |
| `infer_file(input_path, output_path)` | Convert a single audio file |
| `infer_dir(input_dir, output_dir)` | Convert all audio files in a directory |
| `list_models()` | Return a list of available model names |
| `set_params(**kwargs)` | Set inference parameters (see Parameters section) |
| `set_device(device)` | Change the computation device at runtime |
| `set_models_dir(path)` | Change the models directory and reload the model list |

### Legacy Functions

For simpler one-shot usage, `rvc_python.infer_old` provides standalone functions:

```python
from rvc_python.infer_old import infer_file, infer_files

# Convert a single file
infer_file(
    input_path="input.wav",
    model_path="model.pth",
    index_path="model.index",
    device="cuda:0",
    f0method="rmvpe",
    opt_path="output.wav",
    f0up_key=2
)

# Batch convert files
infer_files(
    dir_path="input_dir/",
    model_path="model.pth",
    device="cuda:0",
    opt_dir="output_dir/",
    out_format="wav"
)
```

---

## Model Management

### Directory Structure

Models are organized in subdirectories under the `models_dir`:

```
rvc_models/
├── my_singing_voice/
│   ├── my_singing_voice.pth      # Required: model weights
│   └── my_singing_voice.index    # Optional: FAISS feature index
├── speech_clone/
│   └── speech_clone.pth          # Works without index too
└── another_voice/
    ├── model.pth
    └── trained_index.index
```

### Model Formats

- **`.pth`** (required): PyTorch checkpoint containing model weights, config, and metadata. This is the primary model file.
- **`.index`** (optional): A FAISS index file containing pre-computed speaker feature vectors. When provided, the index is used during inference to retrieve similar features, improving voice similarity and conversion quality. Set `index_rate > 0` to enable index-based retrieval.

### Model Versions

| Version | Feature Dim | Hidden Channels | Description |
|---------|-------------|-----------------|-------------|
| v1 | 256 | 192 | Original RVC models |
| v2 | 768 | 256 | Improved RVC models with better quality |

The version is auto-detected from the model checkpoint in most cases, but you can override it with the `--version` / `version` parameter.

---

## Pitch Extraction Methods

RVC-Python supports four pitch extraction algorithms, each with different trade-offs:

| Method | Flag | Speed | Quality | GPU Required | Notes |
|--------|------|-------|---------|-------------|-------|
| **RMVPE** | `rmvpe` | Fast | Excellent | Optional (recommended) | Best overall. Built-in model, no external dependency. |
| **Harvest** | `harvest` | Slow | Good | No | Uses PYWORLD. High accuracy but slow on CPU. |
| **Crepe** | `crepe` | Medium | Very Good | Yes | Uses torchcrepe. Requires GPU for reasonable speed. |
| **PM** | `pm` | Fast | Good | No | Uses Parselmouth (Praat). Fast and lightweight. |

**Recommendation**: Use `rmvpe` for the best balance of speed and quality. It is the default and recommended method.

---

## Configuration & Parameters

### CLI Parameters

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | `str` | — | Path to input audio file |
| `--dir` | `-d` | `str` | — | Path to input directory for batch processing |
| `--output` | `-o` | `str` | `"out.wav"` | Output file path or directory |
| `--model` | `-mp` | `str` | — | Path to `.pth` model file (required for CLI) |
| `--index` | `-ip` | `str` | `""` | Path to `.index` file |
| `--device` | `-de` | `str` | `"cpu:0"` | Computation device |
| `--method` | `-me` | `str` | `"rmvpe"` | Pitch extraction method (`harvest`, `crepe`, `rmvpe`, `pm`) |
| `--version` | `-v` | `str` | `"v2"` | Model version (`v1`, `v2`) |
| `--index_rate` | `-ir` | `float` | `0.6` | Feature search ratio (0.0–1.0) |
| `--filter_radius` | `-fr` | `int` | `3` | Median filter radius for pitch |
| `--resample_sr` | `-rsr` | `int` | `0` | Output sample rate (0 = keep model SR) |
| `--rms_mix_rate` | `-rmr` | `float` | `0.25` | Volume envelope mix rate |
| `--protect` | `-pr` | `float` | `0.5` | Voiceless consonant protection |
| `--pitch` | `-pi` | `int` | `0` | Pitch shift in semitones |
| `--port` | `-p` | `int` | `5050` | API server port |
| `--listen` | `-l` | flag | — | Listen on all interfaces |
| `--models_dir` | `-md` | `str` | `"rvc_models"` | Models directory |
| `--preload-model` | `-pm` | `str` | — | Preload model on API startup |

### Python Parameters

When using `rvc.set_params()` or constructing the inference class, these parameters are available:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `f0method` | `str` | `"harvest"` | `rmvpe`, `harvest`, `crepe`, `pm` | Pitch extraction algorithm |
| `f0up_key` | `int` | `0` | -12 to 12 | Pitch shift in semitones (positive = higher) |
| `index_rate` | `float` | `0.5` | 0.0–1.0 | How much to rely on the index features. 0 = ignore index, 1 = fully use index. |
| `filter_radius` | `int` | `3` | 0–7 | Median filtering applied to the pitch contour. Higher = smoother but less responsive. |
| `resample_sr` | `int` | `0` | 16000+ | Output sample rate. 0 keeps the model's native sample rate (32k/40k/48k). |
| `rms_mix_rate` | `float` | `1` | 0.0–1.0 | How much of the original audio's volume envelope to preserve. 1 = full original envelope. |
| `protect` | `float` | `0.33` | 0.0–0.5 | Protection for voiceless consonants and breath sounds. Higher = more protection. |

---

## Advanced Topics

### Performance Optimization

1. **Use RMVPE pitch extraction**: It is significantly faster than Harvest while maintaining excellent quality.
2. **Enable half-precision**: Automatically enabled on GPUs with >4GB VRAM. The config detects GPU memory and adjusts accordingly.
3. **Use an index file**: Setting `index_rate` to 0.5–0.8 with a trained index significantly improves voice similarity.
4. **Batch processing**: Use `infer_dir()` instead of looping `infer_file()` to avoid repeated model loading overhead.
5. **GPU selection**: On multi-GPU systems, specify the GPU with `cuda:N` (e.g., `cuda:1`).

### Batch Processing

```python
from rvc_python.infer import RVCInference

rvc = RVCInference(device="cuda:0")
rvc.load_model("voice_model")

# Process an entire directory
results = rvc.infer_dir(
    input_dir="raw_recordings/",
    output_dir="converted/"
)
print(f"Converted {len(results)} files")
```

### ONNX Export

Export a loaded model to ONNX format for deployment in non-Python environments:

```python
from rvc_python.modules.onnx.export import export_onnx

export_onnx(
    ModelPath="path/to/model.pth",
    ExportedPath="path/to/model.onnx"
)
```

### TorchScript JIT Compilation

Compile the RMVPE or synthesizer model using TorchScript for faster inference:

```python
from rvc_python.lib.jit import rmvpe_jit_export, synthesizer_jit_export

# Export RMVPE as TorchScript
rmvpe_jit_export(
    model_path="base_model/rmvpe.pt",
    mode="script",
    device=torch.device("cuda:0"),
    is_half=True
)

# Export synthesizer as TorchScript
synthesizer_jit_export(
    model_path="model.pth",
    mode="script",
    device=torch.device("cuda:0"),
    is_half=True
)
```

### DirectML Support (Windows)

For AMD or Intel GPUs on Windows, DirectML can be used via ONNX Runtime:

1. Install `onnxruntime-directml` and `torch-directml`
2. Set `is_dml=True` in the Config or use the `device` string `"privateuseone:0"`
3. The pipeline will automatically use the DirectML execution provider for RMVPE inference

### IPEX Support (Intel GPUs)

For Intel Arc GPUs and data center GPUs, the library includes an IPEX integration module that automatically hijacks CUDA calls to use Intel XPU:

```python
# If IPEX is installed, it is auto-detected in config.py
# The library will automatically use XPU when available
rvc = RVCInference(device="xpu:0")
```

The IPEX module monkey-patches `torch.cuda` to redirect to `torch.xpu`, enabling seamless GPU acceleration on Intel hardware.

---

## Architecture

```
rvc_python/
├── __init__.py              # Package entry point
├── __main__.py              # CLI and API entry point
├── infer.py                 # Main RVCInference class
├── infer_old.py             # Legacy standalone functions
├── api.py                   # FastAPI REST server
├── download_model.py        # Base model downloader (HuBERT, RMVPE)
├── configs/
│   ├── config.py            # Device detection, GPU config, singleton Config
│   ├── v1/                  # V1 model configs (32k, 40k, 48k)
│   └── v2/                  # V2 model configs (32k, 48k)
├── lib/
│   ├── audio.py             # Audio I/O (load, wav2 via PyAV)
│   ├── rmvpe.py             # RMVPE pitch extraction model
│   ├── slicer2.py           # Audio slicing utility
│   ├── globals/             # Global state
│   ├── jit/                 # TorchScript export/load utilities
│   └── infer_pack/          # VITS model architectures
│       ├── models.py        # Synthesizer models (256/768, F0/no-F0)
│       ├── models_dml.py    # DirectML-compatible model variants
│       ├── models_onnx.py   # ONNX-compatible model variants
│       ├── onnx_inference.py # ONNX inference wrapper
│       ├── attentions.py    # Multi-head attention, FFN, encoder/decoder
│       ├── modules.py       # WaveNet, ResBlock, flow layers
│       ├── transforms.py    # Rational quadratic spline transforms
│       ├── commons.py       # Shared utilities (padding, masking)
│       └── modules/F0Predictor/ # Standalone F0 predictors
└── modules/
    ├── vc/
    │   ├── modules.py       # VC class (model loading, inference)
    │   ├── pipeline.py      # Full pipeline (F0, feature extraction, synthesis)
    │   └── utils.py         # HuBERT loader, index utilities
    ├── ipex/                # Intel XPU hijacking layer
    ├── train/
    │   └── preprocess.py    # Audio preprocessing for training
    └── onnx/
        └── export.py        # ONNX export utility
```

### Inference Pipeline

```
Input Audio
    │
    ▼
┌──────────────────┐
│  Audio Loading   │  (PyAV / librosa)
│  & Resampling    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  HuBERT          │  (fairseq) → 256-dim / 768-dim features
│  Feature Extract │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  FAISS Index     │  (optional) → weighted feature retrieval
│  Lookup          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Pitch Detection │  (RMVPE / Harvest / Crepe / PM)
│  (F0)            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  VITS Synthesis  │  (Generator + Flow + Posterior Encoder)
│  + F0 Condition  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Post-Processing │  (RMS mixing, resampling, normalization)
└────────┬─────────┘
         │
         ▼
    Output Audio
```

---

## Troubleshooting

### Common Issues

#### `torch.load` security warning

In Python 3.12+, PyTorch defaults `weights_only=True` for `torch.load()`. This library explicitly sets `weights_only=False` for RVC model checkpoints, which require pickle deserialization. Only load models from trusted sources.

#### `pyworld` import error

If you see an error about `pyworld` not being available, install it:

```bash
pip install pyworld
```

This is only needed for the `harvest` pitch extraction method. If you use `rmvpe` (the default and recommended method), you don't need pyworld.

#### `torchcrepe` import error

Similarly, `torchcrepe` is only needed for the `crepe` pitch method:

```bash
pip install torchcrepe
```

#### CUDA out of memory

- Reduce batch size or audio length
- Use CPU: `device="cpu:0"`
- The library automatically falls back to FP32 on GPUs with less than 4GB VRAM
- Try a model with a lower sample rate (32k instead of 48k)

#### `faiss-cpu` installation failure on Python 3.12+

The dependency specification now requires `faiss-cpu>=1.8.0` which has proper Python 3.12+ wheels. If you encounter issues:

```bash
pip install faiss-cpu --upgrade
```

#### `fairseq` installation issues

The library uses a fork of fairseq (`fumiama/fairseq`). If installation fails:

```bash
pip install git+https://github.com/fumiama/fairseq.git --no-build-isolation
```

#### FFmpeg not found

Audio loading requires FFmpeg. Install it via your system package manager:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

#### Intel XPU / IPEX not detected

Ensure `intel_extension_for_pytorch` is installed and your Intel GPU drivers are up to date. The IPEX integration is optional and only activates when both IPEX and an XPU device are detected.

---

## Changelog

### v0.2.0

- **Python 3.12+ support**: Updated minimum Python version to 3.12, fixed all compatibility issues
- **`torch.load()` fixes**: Added `weights_only=False` to all model loading calls for Python 3.12+ compatibility
- **Relaxed numpy constraint**: Removed `numpy<2.0.0` upper bound to support numpy 2.x
- **Updated numba**: Bumped minimum to `>=0.60.0` for Python 3.12+ support
- **Updated faiss-cpu**: Now requires `>=1.8.0` with proper Python 3.12+ wheels
- **Graceful pitch imports**: `pyworld`, `torchcrepe`, and `parselmouth` imports are now optional with clear error messages
- **Fixed device check logic**: Corrected `or` to `and` in MPS/XPU device type casting
- **Added missing dependencies**: `fastapi`, `uvicorn`, `loguru`, `pyworld`, `torchcrepe`, `av`
- **Comprehensive documentation**: Full README with architecture, API reference, and troubleshooting
- **Bug fix**: Added missing `os` import in `modules.py`

### v0.1.5

- Initial release from upstream

---

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes and add tests
4. Commit with descriptive messages: `git commit -m "Add new feature"`
5. Push to your fork: `git push origin feature/my-feature`
6. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/onxlmao/rvc-python.git
cd rvc-python
pip install -e .
```

### Reporting Issues

When reporting bugs, please include:
- Python version (`python --version`)
- Operating system and architecture
- PyTorch version (`torch.__version__`)
- GPU model and driver version (if applicable)
- Full error traceback
- Steps to reproduce

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
