# MoooWind AI

MoooWind is a local PyQt5 AI assistant. It can use locally cached language,
vision, and image-generation models; model weights are not included in this
repository.

## Requirements

- Windows 10/11
- Python 3.10 or newer
- NVIDIA GPU with a compatible CUDA driver for the 4-bit text and vision models
- Several dozen gigabytes of free disk space for all model snapshots

Install a CUDA-enabled PyTorch build that matches your NVIDIA driver, then
install the application dependencies in the same Python environment:

```powershell
python -m pip install PyQt5 transformers accelerate bitsandbytes psutil pandas python-docx python-pptx pypdf Pillow
```

Optional features have additional dependencies:

```powershell
python -m pip install -r requirements-book-library.txt
python -m pip install -r requirements-feature-modules.txt
```

AkShare requires an internet connection when online stock data is selected.
The application does not use a hosted language-model API.

## Download models

Run these commands from the project directory. The download script places
snapshots under the local `models` directory, which is intentionally excluded
from Git:

```powershell
python -m pip install modelscope
python download_all_models.py
```

The script downloads the following model snapshots:

| Model | Feature |
| --- | --- |
| `Qwen/Qwen2.5-7B-Instruct` | Companion chat and Office content |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Work mode |
| `Qwen/Qwen2.5-Coder-3B-Instruct` | Code mode |
| `Qwen/Qwen2.5-1.5B-Instruct` | Small-model mode |
| `Qwen/Qwen2.5-VL-3B-Instruct` | Image, video, and handwriting recognition |
| `AI-ModelScope/sd-turbo` | Local image generation |

MoooWind reads models from the local cache only. It reports an error if a
required snapshot is missing rather than downloading model files at runtime.

## Run

```powershell
python Mooowind_App.py
```

## Tests

Run the test suite from the project directory:

```powershell
python -m unittest discover -s tests -v
```

Tests that create Qt widgets use the offscreen platform when needed:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -s tests -v
```

## Local data and generated files

Model weights, chat memories, task checkpoints, MNIST data, Python caches, and
generated output files are excluded from Git by `.gitignore`. Do not commit
personal conversation memories or generated documents.

## License

No license is currently included. Until a license is added, standard copyright
applies and reuse permissions are not granted by this repository.
