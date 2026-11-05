# torch-directml-backwards

## Overview

An unofficial maintenance fork of [torch-directml](https://pypi.org/project/torch-directml/), currently under development.

## Compatibility

Windows x64, standard CPython with the GIL enabled, and DirectML 1.15.4.

| CPython | PyTorch 2.14.1 | PyTorch 2.15.0 |
| --- | --- | --- |
| 3.12 | Supported | Supported |
| 3.13 | Supported | Supported |
| 3.14 | Supported | Supported |
| 3.15 | Not supported | Supported |

GPU tests used an NVIDIA Tesla T4; other hardware remains unverified. See [validation details](docs/native-abi.md#runtime-checks).

## Quick Start

Use [Python](https://www.python.org/downloads/windows/), a DirectX 12 GPU driver and the [Microsoft Visual C++ x64 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

For Python 3.12 and PyTorch 2.15.0, run the following from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install .\dist\cp312-torch2.15.0-win_amd64\torch_directml_backwards-0.1.0.dev261106-2.15.0-cp312-cp312-win_amd64.whl
```

See [installation details](docs/packaging.md#install-update-and-remove) for other versions, updates and removal, or the [build instructions](docs/development.md#build-a-wheel) to build from source.

## Usage

Move tensors or models to the device returned by `torch_directml.device()`:

```python
import torch
import torch_directml

device = torch_directml.device()
x = torch.tensor([1.0, 2.0]).to(device)
print((x + x).cpu())
```

## Testing

From the repository root, with the built wheel installed:

```powershell
.venv\Scripts\python.exe tests\check_backend.py
.venv\Scripts\python.exe tests\check_amp.py
```

GPU checks run by default. Add `--cpu-only` to run CPU Check only. See [test details](docs/native-abi.md#runtime-checks).

## License

The project's source code is licensed under MIT, except for third-party code identified in [NOTICE](NOTICE). See [LICENSE](LICENSE). The bundled DirectML runtime has [separate license terms](docs/packaging.md#licenses).
