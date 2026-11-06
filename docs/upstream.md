# Upstream source

This project is based on the Windows x64 wheel of [torch-directml 0.2.5.dev240914](https://pypi.org/project/torch-directml/0.2.5.dev240914/) for CPython 3.12.

```text
torch_directml-0.2.5.dev240914-cp312-cp312-win_amd64.whl
```

The Python files in `src/torch_directml/` come from this wheel. The public API structure is preserved, and AMP state is managed by the target PyTorch version. `LICENSE` retains the upstream license text and adds a copyright notice for Cyrne1's modifications. `NOTICE` retains the original text and adds license notices for the PyTorch headers and pybind11 GIL-management code used by the added implementation.

This project's version is independent of the upstream version. It targets standard CPython builds with the GIL enabled: PyTorch 2.14.1 on Python 3.12 through 3.14, and PyTorch 2.15.0 on Python 3.12 through 3.15. The development environment uses PyTorch 2.15.0; each distribution wheel pins the PyTorch version it was compiled for. [torchvision 0.29.2](https://pypi.org/project/torchvision/0.29.2/) is used because its official wheel is independent of the Python version.

The original native extension and DirectML 1.13.1 DLL are retained in the upstream wheel. The [native modifications](native-abi.md) apply ABI updates and interpolation-coordinate/cache-key fixes to the existing `.pyd`. The original operator execution paths and Python API are preserved.

Distribution wheels use the official [Microsoft.AI.DirectML 1.15.4](https://www.nuget.org/packages/Microsoft.AI.DirectML/1.15.4) NuGet package. The DLL is included without modification, along with its license and third-party notices. Its terms are separate from the source code's MIT license. See the [native implementation](native-abi.md) for the checked functionality and [wheel packaging](packaging.md) for the runtime source and packaging procedure.

## Code conventions

Code comments and docstrings are written in English. Python naming, API order, indentation and line endings follow `torch_directml/` in the upstream wheel. Added Python code uses `snake_case` for functions and variables, `PascalCase` for classes, and four-space indentation.

The upstream wheel does not contain the native extension's C++ source. C++ conventions follow the patterns shared by the upstream repository's public [Python bindings](https://github.com/microsoft/DirectML/blob/8700779fe7a09ea7a007cf3d7ab4293c78e41017/Python/src/device.h), [utilities](https://github.com/microsoft/DirectML/blob/8700779fe7a09ea7a007cf3d7ab4293c78e41017/Python/src/util.h) and [queue management](https://github.com/microsoft/DirectML/blob/8700779fe7a09ea7a007cf3d7ab4293c78e41017/DxDispatch/src/dxdispatch/Device.h). These conventions are inferred from public code; they have not been confirmed as an exact match for the unpublished source.

- Use four-space indentation. Place opening braces on a new line for functions, classes, namespaces and multiline control blocks.
- Use `PascalCase` for types and ordinary functions, `camelCase` for variables, and `m_` followed by `camelCase` for member variables. Place public methods before private implementation and members.
- Use the backend namespace and Tensor/guard type names found in the original extension's RTTI: `torch_dml`, `DmlTensorImpl` and `DmlGuardImpl`.

Preserve names required by callers, including PyTorch override methods, macro-required names and C exports. Also preserve field order, types and calling conventions that form part of the ABI.
