# Development tools

Run commands from the repository root. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.

## Development versions

Use `X.Y.Z.devYYMMDD` for development releases, where `YYMMDD` is the release date. Package versions have no `v` prefix; Git tags use `vX.Y.Z.devYYMMDD`.

## Python tools

```sh
uv sync --python 3.12 --locked --only-group dev
```

This creates `.venv` with Python 3.12 and the development tools from `uv.lock`. It does not install the project's runtime dependencies.

| Tool | Purpose |
| --- | --- |
| LIEF | Read and modify PE files, including DLL imports and exports |
| Capstone | Disassemble machine instructions |
| wheel | Assemble and inspect wheel distributions |
| Ruff | Check Python code |

Use `.venv/bin/python` on Linux or `.venv/Scripts/python.exe` on Windows to run the build tools.

## Native build tools

Native compilation requires `clang-cl`, `lld-link`, and Microsoft CRT/Windows SDK headers and libraries. On Windows, install Visual Studio 2022 17.1 or newer with the C++ desktop tools and Windows SDK, plus LLVM 20 or newer. Use an x64 developer command prompt with `clang-cl` and `lld-link` on `PATH`. On Linux, obtain the headers and libraries with [xwin](https://github.com/Jake-Shadle/xwin) and pass the extracted directory with `--windows-sdk`. The examples use `.tools/msvc`.

## Build a wheel

The build command downloads the official upstream wheel, target PyTorch wheel, CPython SDK and DirectML NuGet package. Their SHA256 hashes are checked against the repository's profiles. It then runs the existing native compiler, binary patcher and wheel packager.

```sh
.venv/bin/python tools/build.py matrix
.venv/bin/python tools/build.py build cp312 2.15.0 --windows-sdk .tools/msvc --output-dir .work/dist/cp312-torch2.15.0-win_amd64
```

This example is for Linux. On Windows, use `.venv\Scripts\python.exe` and omit `--windows-sdk` when the developer environment is active. Outputs go to `.work/cp312-torch2.15.0-win_amd64/build/` and `.work/dist/cp312-torch2.15.0-win_amd64/`, leaving the distributed wheels in `dist/` untouched. Existing build or output directories are rejected; choose new locations with `--work-dir` and `--output-dir` when retaining an earlier build.

To build all supported targets on Windows, run `tools\ci\build.ps1 -Revision local` after creating the development environment. Then run `tools\ci\cpu.ps1 -Revision local` for CPU Check. The build initializes the Visual Studio environment and writes wheels to `.work/ci/artifacts/`; CPU Check installs each generated wheel into a separate virtual environment. Both scripts accept `-Abis cp313`, for example, to limit the targets to one Python ABI. The target list comes from `native/profiles/cp*.toml`; it is not maintained as a second CI matrix.

## Updating dependencies

- Upstream torch-directml: update the source URL, SHA256 and native member in `native/profiles/common.toml`. Recheck every binary edit against the new original; do not carry RVAs over without inspecting the binary.
- PyTorch: add the official Windows wheel hash to each supported CPython profile. Review changed C++ APIs and layouts before building. Update the development default in `pyproject.toml` and run `uv lock` if needed.
- CPython: update the SDK version and hash in the corresponding profile. A new minor version also needs its own ABI profile and native compatibility review. Free-threaded builds are outside the current profiles.
- DirectML: update the NuGet version and SHA256 in `pyproject.toml`. Keep its DLL and license files unchanged.

For an updated target, install the generated wheel into a fresh environment and run the [runtime checks](native-abi.md#runtime-checks). Changes shared by all targets require checks across the full matrix. Record the actual compiler, SDK, Python, PyTorch and GPU driver versions with the results. A build or resolved import table alone is not a runtime check.

See the [native build and test commands](native-abi.md#build) and [wheel packaging](packaging.md).
