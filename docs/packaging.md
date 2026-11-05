# Wheel packaging

`tools/build_wheel.py` assembles a Windows x64 wheel from the Python sources, the official DirectML runtime, and the modified native extension. It identifies the Python ABI from the actual Python runtime DLL import and checks it against the available profiles. It also checks the PE format, `PyInit_torch_directml_native` export, and compiled PyTorch version. These checks do not establish GPU correctness.

Download [Microsoft.AI.DirectML 1.15.4](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4) and save the original NuGet package in `reference/directml/`. Its version and SHA256 are pinned in `pyproject.toml` under `[tool.torch-directml]`. Packaging verifies this hash and copies `bin/x64-win/DirectML.dll` without modification; it does not use the older runtime bundled in the upstream torch-directml wheel.

```sh
.venv/bin/python tools/build_wheel.py \
  --native .work/cp315-torch2.15.0-win_amd64/build/torch_directml_native.cp315-win_amd64.pyd \
  --directml-package reference/directml/microsoft.ai.directml.1.15.4.nupkg \
  --output-dir .work/dist/cp315-torch2.15.0-win_amd64
```

The example produces a `cp315-cp315-win_amd64` wheel in `.work/dist/cp315-torch2.15.0-win_amd64/`, with `Requires-Python: >=3.15,<3.16`. The 3.12 through 3.14 extensions likewise produce wheels restricted to their own Python minor version. The target is independent of the Python running the packaging tool. The checked-in `dist/` directory contains the distributed wheels.

The wheel build tag identifies its PyTorch version, for example `torch_directml_backwards-0.1.0.dev261106-2.15.0-cp315-cp315-win_amd64.whl`. The same value is recorded in `WHEEL`, and the wheel's metadata pins that PyTorch version.

An existing wheel with the same name is left untouched and causes an error.

The wheel contains the Python sources from `src/torch_directml/` and the modified native extension as a top-level module, matching upstream's `import torch_directml_native`. `native-code.bin` is a linker intermediate used during extension patching and is excluded from the wheel. `wheel.wheelfile.WheelFile` writes the distribution's `RECORD` hashes.

The modified `.pyd` contains the replacement code and identifies its target through the `tdml_torch_version` export. Packaging checks that this version is accepted by the ABI profile and pins the wheel's `torch` dependency to it. The development environment's default PyTorch version does not change which version a built wheel requires.

## Install, update and remove

Use Windows x64, a supported standard CPython with the GIL enabled, a DirectX 12 GPU driver and the [Microsoft Visual C++ x64 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). Real-GPU checks currently cover Windows Server 2025 with an NVIDIA T4; other Windows/GPU combinations are not verified by that result. Match the wheel's `cp312`, `cp313`, `cp314` or `cp315` tag to the Python minor version.

In PowerShell, create an environment with the intended Python version, then install the selected wheel. For Python 3.12 and PyTorch 2.15.0:

```powershell
python --version
python -m venv .venv
.venv\Scripts\python.exe -m pip install .\torch_directml_backwards-0.1.0.dev261106-2.15.0-cp312-cp312-win_amd64.whl
.venv\Scripts\python.exe -m pip check
```

The package requires the compiled PyTorch version and torchvision version in its metadata; pip resolves them during installation. The reviewed target wheels are the official PyPI Windows distributions. Other PyTorch build variants have not been checked. Source-tree `pip install .` does not build the native extension; use the build and wheel packaging commands instead.

To replace an installed build with another wheel of the same project version, use `pip install --force-reinstall` with the selected filename. The command also installs that wheel's required PyTorch version.

The upstream `torch-directml` package and this package share import paths and native filenames. Uninstall the upstream package before installing this fork; do not keep both in the same environment. If both were installed, uninstall both packages and reinstall the selected fork wheel to restore its files.

To remove the fork, run `.venv\Scripts\python.exe -m pip uninstall torch-directml-backwards`. Shared dependencies such as PyTorch remain installed.

## CI artifacts

The [CI workflow](../.github/workflows/ci.yml) provides the tested wheels and a manifest listing the GPU checks performed. Commit CI checks the affected scope; these artifacts do not imply complete release validation.

## Licenses

The project's source code remains MIT-licensed, except for third-party code identified in `NOTICE`. Each wheel contains `LICENSE` and `NOTICE` under its `.dist-info/licenses/` directory, plus the runtime's original `LICENSE.txt` and `ThirdPartyNotices.txt` under `.dist-info/licenses/directml/`. `NOTICE` preserves the upstream notices and includes the PyTorch and pybind11 BSD-3-Clause terms. The runtime is distributed under the Microsoft DirectML license, not MIT; `project.license` and the wheel's `License-Expression` both declare `MIT AND BSD-3-Clause AND Apache-2.0 AND LicenseRef-Microsoft-DirectML` for the bundled distribution. The wheel lists all four license files in its metadata.
