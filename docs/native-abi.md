# Native implementation

The current targets are PyTorch 2.14.1 on CPython 3.12 through 3.14 and PyTorch 2.15.0 on CPython 3.12 through 3.15, Windows x64, with the GIL enabled. They use the native extension from `torch-directml 0.2.5.dev240914` as their source. Changing the wheel tag alone does not make an extension compatible with another Python ABI.

The modified extension contains the original DirectML operator code and replacement C++ code compiled against the target PyTorch headers. Calls to unchanged PyTorch APIs remain direct DLL imports; calls to replaced functions go to code inside the `.pyd`. No separate compatibility DLL is used. The upstream Python API is preserved; [wheel packaging](packaging.md) supplies the unmodified official DirectML 1.15.4 runtime.

## Changes at the PyTorch boundary

| Area | Change |
| --- | --- |
| Imports | Update renamed exports and replace removed entry points inside the extension. |
| Dispatch | Update PrivateUse1, AutogradPrivateUse1 and SparsePrivateUse1 registration values and the PrivateUse1 key-set mask. |
| Tensor | Use the current `TensorImpl` base and virtual interface, preserving the original DirectML handle at `this + 0xb8`. Keep the DirectML custom size/stride policy after detach. |
| Registration | The original initializer allocates 32-byte kernel records and 96-byte function records. Construct current 40-byte kernels and 104-byte functions at registration, transferring ownership through current constructors. |
| String arguments | Match the five original string-view signature types to `std::string_view`. Their Windows pointer/length layout and calling convention match. |
| Device guard | Register the current interface and the existing `privateuseone` backend name. Update old virtual forwarding calls to the current method slots. |
| Generator | Use current hooks and the current CPU generator vtable, preserving the backend's CPU-based random generation. |
| Logger | Retain the original inline stream layout and emit its message through the current PyTorch logger. |
| CPU fallback | Supply the additional CPU dispatch-key argument from a small function inside the original text section. |
| Undefined Tensor | Replace the imported singleton data access with a call to the current getter. |
| Named tensors | Remove obsolete calls because the target PyTorch has no named-tensor state. |
| NMS | Call the current typed dispatcher, preserving CPU NMS computation and the original output device. |
| Resampling | Match bilinear half-pixel coordinates in forward and backward, and retain fractional parameters in compiled-operator cache keys. |

The reviewed instruction changes are shared by PyTorch 2.14.1 and 2.15.0. The profile selects the target by its wheel hash, and the replacement code is compiled against that wheel's headers and import libraries. Rebuild for each target; sharing these instruction changes does not make the compiled extension independent of the PyTorch version.

The initialization-state hook reads the original backend manager's `INIT_ONCE` flags without initializing a GPU. Its addresses, the retained DirectML handle offset, and the original method addresses belong to the pinned source wheel. Changing the source wheel requires reviewing them.

[`native/resample.cpp`](../native/resample.cpp) replaces four offset-array initialization sites inside the original forward and backward functions. The leaf routines use those functions' existing frame pointers and update the same temporary arrays and XMM0 register. They do not add a C++ stack frame, allocate memory, or replace the operator execution path. Their frame offsets and call sites are specific to the pinned source wheel and must be reviewed together when that source changes.

For linear interpolation with `align_corners=False`, the forward H/W input offsets are +0.5 and output offsets are -0.5. The pinned runtime's `ResampleGrad1` requires the opposite offsets, -0.5 and +0.5, for matching backward gradients; passing the forward offsets shifts the gradient contributions in the wrong direction. Batch and channel offsets stay zero. Nearest and `align_corners=True` retain zero offsets.

The compiled-operator cache-key helper copies each float's 32-bit representation into its existing 64-bit slot instead of truncating it to an integer. This distinguishes alignment and explicit-scale settings without changing the cache's storage or lookup mechanism.

## FP16 automatic mixed precision and completion checks

`native/amp.cpp` registers autocast policies for the existing matrix multiplication and convolution paths, with FP32 policies for sensitive reductions, layer/group normalization and losses. It uses PyTorch's casting cache and thread-local autocast state. The Python module's AMP getters and setters refer to that same state. Ordinary FP32 execution remains the default.

```python
import torch
import torch_directml

device = torch_directml.device()
model = torch.nn.Linear(16, 8).to(device)
inputs = torch.randn(4, 16).to(device)
with torch.autocast("privateuseone", dtype=torch.float16):
    output = model(inputs)

torch.accelerator.synchronize(device)
stream = torch.accelerator.current_stream(device)
finished = stream.query()
```

For training, use the standard `torch.amp.GradScaler("privateuseone")`. The scale-update/unscale kernels compose existing DirectML operations; they do not copy gradients to CPU. GradScaler also computes its one-element scale's reciprocal in FP64. DirectML 1.15.4 does not support that reciprocal dtype, so this operation falls back to CPU with a warning; other reciprocal dtypes retain the existing operator path. The composed GradScaler path can be slower than FP32 for small models. FP16 results can differ from FP32, and speedups depend on the GPU and workload. Explicit `dtype`, `out=` and in-place operations retain PyTorch's normal autocast exclusions. Use `binary_cross_entropy_with_logits` instead of `binary_cross_entropy` inside autocast.

Autocast registration is connected explicitly to the original backend initialization. It uses the current C++ function constructor with existing operator schemas rather than introducing schema-inference singletons with separate CRT/TLS initialization. The registration handles live as long as the backend.

`native/synchronization.cpp` reuses the original execution-context mutex, recorder submission and command-queue fence. Completion checks submit commands still being recorded; `query()` reads the fence without waiting for GPU completion. Explicit synchronization waits on a Windows event and checks device removal as well as API/wait failures. Ordinary operators do not gain additional waits. Only the existing default stream is supported; additional streams and GPU timing events remain outside this implementation. Checking an unused device does not initialize it.

The execution-context and queue layouts belong to the pinned upstream wheel. Its uploads, compute and readbacks share this context; changing the source requires checking these layouts and submission paths again. The initialized-device lookup is shared with the existing generator hook.

## Changes at the CPython boundary

The 3.13 through 3.15 profiles update the module's version check and import `PyThreadState_GetUnchecked` in place of the removed private name. The original pybind11 binding code and public function signatures are retained.

Four functions in [`native/python_bindings.cpp`](../native/python_bindings.cpp) handle the changed boundary. GIL acquisition and release follow pybind11 2.12.0's existing thread-state ownership rules, using the target CPython headers for `gilstate_counter`. The functions reuse the original binding manager and TSS keys. GC traversal and clearing use `PyObject_VisitManagedDict` and `PyObject_ClearManagedDict`, avoiding dictionary materialization during traversal. Their implementations are inserted into the `.pyd` through the same mechanism as the PyTorch replacements.

CPython 3.14 adds a tuple hash cache, moving the first element from offset `0x18` to `0x20`. The 3.14 and 3.15 profiles update the original argument dispatcher, tuple result construction and type-base traversal. The list branch retains its separate element pointer at `0x18`; no general replacement of that number is performed.

CPython 3.15 appends a field to `PyTypeObject`, moving the following heap-type fields by eight bytes. Its profile updates the nine name and qualified-name writes in the original three type constructors. Heap types are still allocated by CPython itself; their new allocation size is not hard-coded. The GIL counter also moves, which is handled by recompiling the same four C++ functions against the target headers.

Relevant object, type and code-field accesses were checked against the official 3.12.10, 3.13.17, 3.14.10 and 3.15.2 Windows headers. The 3.12 profile retains its original binding functions.

## Code placement and initialization

The linker reserves the original extension's virtual address range before placing new code. The patcher drops that reservation and appends the compiled sections at their linked addresses. C++ type information and exception metadata therefore retain their linked RVAs.

The patcher combines exception and base-relocation tables, rebuilds imports, and updates code references to their destinations. It preserves the original TLS and initialization sequence. The new module entry initializes the added code's stack-protection cookie before calling the original CRT entry point. Replacement code must not introduce separate TLS or global C++ initializers that require another CRT startup sequence.

Original export names, ordinals and addresses are retained. The added `tdml_torch_version` data export identifies the compiled target. Original section RVAs are preserved; raw file positions may move as the PE headers grow, so debug data's file offsets are updated.

## Build

Set up the [development tools](development.md) and keep the official wheels and CPython SDK here:

```text
reference/torch-directml/torch_directml-0.2.5.dev240914-cp312-cp312-win_amd64.whl
reference/pytorch/torch-2.15.0-cp315-cp315-win_amd64.whl
reference/cpython/3.15.2/python.3.15.2.nupkg
```

[`native/profiles/common.toml`](../native/profiles/common.toml) contains shared instruction changes. [`cp312.toml`](../native/profiles/cp312.toml), [`cp313.toml`](../native/profiles/cp313.toml), [`cp314.toml`](../native/profiles/cp314.toml) and [`cp315.toml`](../native/profiles/cp315.toml) contain ABI-specific changes, accepted wheel hashes and their official CPython SDK NuGet packages. The example uses [Python 3.15.2](https://www.nuget.org/packages/python/3.15.2). On Linux, provide an xwin SDK in `.tools/msvc` and make `clang-cl` and `lld-link` available:

```sh
.venv/bin/python tools/build_native.py \
  reference/torch-directml/torch_directml-0.2.5.dev240914-cp312-cp312-win_amd64.whl \
  reference/pytorch/torch-2.15.0-cp315-cp315-win_amd64.whl \
  --python-sdk reference/cpython/3.15.2/python.3.15.2.nupkg \
  --windows-sdk .tools/msvc

.venv/bin/python tools/native_abi.py patch \
  reference/torch-directml/torch_directml-0.2.5.dev240914-cp312-cp312-win_amd64.whl \
  reference/pytorch/torch-2.15.0-cp315-cp315-win_amd64.whl \
  --python-sdk reference/cpython/3.15.2/python.3.15.2.nupkg \
  --native-code .work/cp315-torch2.15.0-win_amd64/build/native-code.bin \
  --output .work/cp315-torch2.15.0-win_amd64/build/torch_directml_native.cp315-win_amd64.pyd
```

On Windows, use `.venv/Scripts/python.exe` and a developer command prompt with Clang and the Windows SDK; omit `--windows-sdk`. The build extracts headers and import libraries from the selected distributions. For 3.12, select its PyTorch wheel and omit `--python-sdk`. Output directories are separated by Python ABI and PyTorch version.

For PyTorch 2.14.1 on CPython 3.12 through 3.14, select its matching Python wheel and keep the same CPython SDK. Compilation uses the selected wheel's headers and import libraries. CPython 3.15 requires PyTorch 2.15.0: the official 2.14.1 CPU wheel crashes during threaded autograd and process cleanup even without this extension.

`native-code.bin` is an offline linker intermediate, not a runtime module or a distribution artifact. The patcher uses it to update the original `.pyd`. Original wheels remain unchanged, and patching and [packaging](packaging.md) refuse existing output files.

## Runtime checks

With the built wheel installed on Windows, run these scripts from the repository root:

```sh
python tests/check_backend.py
python tests/check_ops.py
python tests/check_nn.py
python tests/check_functions.py
python tests/check_amp.py
```

| Script | Checks |
| --- | --- |
| `check_backend.py` | Import and binding ABI, Tensor lifetime and aliasing, random state, bilinear forward/backward, and synchronization |
| `check_ops.py` | Pointwise, reduction, ordering, indexing and layout operations; integer widths, FP16/FP32, promotion, out/in-place paths, and applicable gradients |
| `check_nn.py` | Matrix products, convolutions, pooling, padding, normalization, losses, gradients, running statistics, and a training step |
| `check_functions.py` | Attention with masks and caches, rotary positions, RMSNorm weights/epsilon, and Phi-2, Llama and Phi-3 MLPs against unfused CPU references |
| `check_amp.py` | Numerical results and dtypes for each explicit autocast policy, context state, exclusions, running statistics, and GradScaler updates |

Numerical checks compare values, shapes, dtypes and devices with CPU references. Integer values and indices must match exactly; floating-point tolerances depend on dtype. Nonuniform upstream gradients check applicable backward paths. These are regression checks, not an exhaustive implementation of every PyTorch operator. Existing CPU fallback warnings remain visible; a successful numerical comparison alone does not establish GPU execution.

For CPU Check without a GPU, run:

```sh
python tests/check_backend.py --cpu-only
python tests/check_amp.py --cpu-only
```

Supported combinations and tested hardware are listed in the [README](../README.md#compatibility). The `torch.accelerator` memory-statistics and cache-control APIs are not implemented.

### Native ABI and queue checks

CI also builds and runs a small C++ checks DLL against each target's PyTorch headers and libraries. It calls the actual fallback, Tensor constructor and NMS code inside the installed extension. NMS uses different thresholds and checks exception propagation. GPU checks block the backend queue with a test-owned fence, then observe its submission fence before any CPU readback to test query and synchronization independently.

To build the same checks on Windows, use the [native build environment](development.md#native-build-tools) with CMake 3.25 or newer and Ninja on `PATH`:

```sh
python tools/build.py build cp312 2.15.0 --native-checks --work-dir .work/check-build --output-dir .work/check-artifacts
```

Install the wheel generated in `.work/check-artifacts/` with the target interpreter, then run:

```sh
python tests/check_backend.py --native-checks .work/check-artifacts/native_checks.dll
```

Add `--cpu-only` to run only the native CPU checks. The adjacent `native_checks.json` contains internal function addresses and the matching extension hash, so LIEF is needed only in the build environment. A mismatched installed extension is rejected before calling those addresses. Without `--native-checks`, the standalone script reports the independent queue/fence check as not run.
