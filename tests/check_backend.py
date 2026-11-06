"""Check the installed extension, Tensor lifetimes, random state and queue completion."""

import argparse
import ctypes
import hashlib
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from math import prod
from pathlib import Path


def load_native_checks(path, native):
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with Path(native.__file__).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != metadata["native_sha256"]:
        raise ValueError("Native checks were built for a different installed extension")
    module = ctypes.WinDLL(native.__file__)
    marker = ctypes.c_char.in_dll(module, "tdml_torch_version")
    symbols = metadata["rvas"]
    if ctypes.addressof(marker) - module._handle != symbols["tdml_torch_version"]:
        raise ValueError("Native check RVAs do not match the installed extension")
    client = ctypes.CDLL(str(path.resolve()))
    client.tdml_check_native.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    client.tdml_check_native.restype = ctypes.c_int
    if client.tdml_check_native(
        module._handle, symbols["tdml_nms"], symbols["tdml_tensor_construct"]
    ):
        raise AssertionError("In-module native checks failed")
    return client, module


def check_synchronization(device):
    import torch
    from common import assert_result, sample

    stream = torch.accelerator.current_stream(device)
    for size in (32, 48, 32):
        source = sample((size, size))
        value = source.to(device)
        result = value @ value.T
        stream.synchronize()
        if not stream.query():
            raise AssertionError(
                "The default stream is incomplete after synchronization"
            )
        assert_result(result, source @ source.T, device, atol=1e-4)
        result = value + size
        torch.accelerator.synchronize(device)
        if not stream.query():
            raise AssertionError("Device synchronization did not complete pending work")
        assert_result(result, source + size, device)

    def work(offset):
        source = sample((64,)) + offset
        value = source.to(device)
        for _ in range(4):
            value = value + 1
            torch.accelerator.synchronize(device)
        return value, source + 4

    with ThreadPoolExecutor(max_workers=2) as pool:
        for actual, expected in pool.map(work, (1, 10)):
            assert_result(actual, expected, device)


def check_resampling(device):
    import torch
    from common import check
    from torch.nn import functional

    # Reuse shapes with different options to expose an incomplete compiled-operator cache key.
    for shape, size, order in (
        ((1, 2, 3, 4), (5, 7), (False, True, False)),
        ((1, 1, 5, 7), (3, 4), (True, False, True)),
        ((1, 1, 2, 3), (1, 1), (False, True)),
        ((1, 1, 1, 3), (3, 6), (True, False)),
    ):
        source = (
            torch.arange(prod(shape), dtype=torch.float32)
            .sin()
            .reshape(shape)
            .requires_grad_()
        )
        for align_corners in order:
            check(
                f"bilinear/{shape}/{size}/align={align_corners}",
                functional.interpolate,
                (source,),
                device,
                gradients=True,
                kwargs={
                    "mode": "bilinear",
                    "size": size,
                    "align_corners": align_corners,
                },
            )
    source = (
        torch.arange(15, dtype=torch.float32).sin().reshape(1, 1, 3, 5).requires_grad_()
    )
    for recompute in (False, True, False):
        check(
            f"bilinear/explicit-scale/recompute={recompute}",
            functional.interpolate,
            (source,),
            device,
            gradients=True,
            kwargs={
                "mode": "bilinear",
                "scale_factor": 1.5,
                "align_corners": False,
                "recompute_scale_factor": recompute,
            },
        )

    # Nearest sampling shares the patched resampling code but must not receive half-pixel offsets.
    for size in ((5, 7), (2, 3)):
        check(
            f"nearest/{size}",
            functional.interpolate,
            (source,),
            device,
            kwargs={"mode": "nearest", "size": size},
            gradients=True,
        )


def check_factories(device):
    import torch
    from common import assert_result

    for dtype in (
        torch.float16,
        torch.float32,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
    ):
        for function, args in (
            (torch.zeros, ((2, 3),)),
            (torch.ones, ((2, 3),)),
            (torch.full, ((2, 3), 3)),
            (torch.eye, (3, 4)),
        ):
            assert_result(
                function(*args, dtype=dtype, device=device),
                function(*args, dtype=dtype, device="cpu"),
                device,
            )
        if dtype is not torch.bool:
            assert_result(
                torch.arange(1, 12, 3, dtype=dtype, device=device),
                torch.arange(1, 12, 3, dtype=dtype),
                device,
            )
    source = torch.tensor([[-2.4, -1.3, 0.0], [0.2, 1.3, 2.4]])
    for dtype in (torch.float16, torch.int16, torch.int32, torch.int64, torch.bool):
        assert_result(source.to(device).to(dtype), source.to(dtype), device)
    # Storage copies must retain FP64 precision even when the GPU lacks FP64 arithmetic.
    double = torch.tensor([2**32 + 1.25, -(2**32) - 0.125], dtype=torch.float64)
    assert_result(double.to(device).clone(), double, device, rtol=0, atol=0)
    print("PASS tensor factories and dtype conversions", flush=True)


def check_lifetimes(device):
    import torch
    from common import assert_result, sample
    from torch.nn import functional

    source = sample((3, 4))
    original = source.to(device)
    detached = original.detach()
    # Read opaque Tensor metadata before CPU readback can materialize a contiguous copy.
    for value in (original, detached):
        if value.stride() != source.stride() or not value.is_contiguous():
            raise AssertionError(
                "A contiguous Tensor lost its strides after transfer or detach"
            )
    original.add_(0.25)
    assert_result(detached, source + 0.25, device)
    view = detached.T
    del original, detached
    # Allocate other handles after the source dies, before reading the retained view.
    for _ in range(8):
        temporary = torch.ones((3, 4), device=device) * 7
        del temporary
    assert_result(view, (source + 0.25).T, device)
    assert_result(view.clone(), (source + 0.25).T, device)
    try:
        view.untyped_storage()
    except RuntimeError as error:
        if "storage" not in str(error).lower():
            raise
    else:
        raise AssertionError("An opaque DirectML Tensor exposed ordinary storage")

    state = torch.random.get_rng_state()
    try:
        shape = (17, 11)
        probabilities = torch.linspace(0.0, 1.0, 11).expand(shape).to(device)
        weights = torch.tensor([0.0, 1.0, 4.0, 0.0, 2.0], device=device)
        for name, generate, expected_shape, dtype in (
            ("rand", lambda: torch.rand(shape, device=device), shape, torch.float32),
            ("randn", lambda: torch.randn(shape, device=device), shape, torch.float32),
            (
                "randint",
                lambda: torch.randint(-7, 11, shape, device=device),
                shape,
                torch.int64,
            ),
            ("randperm", lambda: torch.randperm(17, device=device), (17,), torch.int64),
            (
                "multinomial",
                lambda: torch.multinomial(weights, 17, replacement=True),
                (17,),
                torch.int64,
            ),
            ("bernoulli", lambda: torch.bernoulli(probabilities), shape, torch.float32),
            (
                "dropout",
                lambda: functional.dropout(
                    torch.ones(shape, device=device), p=0.25, training=True
                ),
                shape,
                torch.float32,
            ),
        ):
            torch.privateuseone.manual_seed_all(123)
            initial = torch.random.get_rng_state()
            first = generate()
            second = generate()
            torch.random.set_rng_state(initial)
            repeated = generate()
            if first.shape != expected_shape or first.dtype is not dtype:
                raise AssertionError(f"{name} returned the wrong shape or dtype")
            assert_result(repeated, first.cpu(), device, rtol=0, atol=0)
            if torch.equal(first.cpu(), second.cpu()):
                raise AssertionError(f"{name} did not advance the generator")
            if name == "rand" and not ((first >= 0) & (first < 1)).all().item():
                raise AssertionError("Uniform random values are outside [0, 1)")
            if name == "randint" and not ((first >= -7) & (first < 11)).all().item():
                raise AssertionError(
                    "Integer random values are outside the requested bounds"
                )
            if name == "randperm":
                assert_result(first.sort().values, torch.arange(17), device)
            if (
                name == "multinomial"
                and not ((first == 1) | (first == 2) | (first == 4)).all().item()
            ):
                raise AssertionError(
                    "Multinomial sampled an index with zero probability"
                )
            if name == "bernoulli":
                if not ((first == 0) | (first == 1)).all().item():
                    raise AssertionError("Bernoulli values must be zero or one")
                assert_result(first[:, 0], torch.zeros(17), device)
                assert_result(first[:, -1], torch.ones(17), device)
            if name == "dropout" and not ((first == 0) | (first == 4 / 3)).all().item():
                raise AssertionError(
                    "Dropout did not use the requested keep probability"
                )
    finally:
        torch.random.set_rng_state(state)
    print(
        "PASS retained Tensor handles, detach aliasing, opaque storage and random-state replay"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run CPU checks without a DirectML GPU"
    )
    parser.add_argument(
        "--native-checks",
        type=Path,
        help="Matching C++ checks DLL and JSON from build.py --native-checks",
    )
    args = parser.parse_args()
    import torch

    baseline = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    before = baseline + 1
    if args.native_checks:
        # The in-module NMS bridge needs torchvision's CPU schema and kernel.
        importlib.import_module("torchvision")
    import torch_directml as dml

    # The wrapper must load the bundled DirectML DLL before the extension is imported.
    extension = importlib.import_module("torch_directml_native")
    torch.testing.assert_close(baseline + 1, before)
    if extension.custom_device(0) != torch.device("privateuseone", 0):
        raise AssertionError("Native device binding returned a different device")
    try:
        extension.custom_device("invalid")
    except TypeError:
        pass
    else:
        raise AssertionError("Native device binding accepted an invalid argument")
    for operator in ("aten::add.out", "aten::_copy_from"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(operator, "PrivateUse1"):
            raise AssertionError(f"DirectML kernel was not registered: {operator}")
    generator = torch.Generator(device="privateuseone:0").manual_seed(123)
    if generator.initial_seed() != 123:
        raise AssertionError("Native generator did not retain its seed")
    torch.testing.assert_close(
        generator.clone_state().get_state(), generator.get_state()
    )
    native_checks = (
        load_native_checks(args.native_checks, extension)
        if args.native_checks
        else None
    )
    print(
        f"PASS import, device binding, registration and generator on PyTorch {torch.__version__}"
    )
    if args.cpu_only:
        print("GPU numerical checks not run (--cpu-only)")
        return
    if not dml.is_available():
        raise RuntimeError("A DirectML GPU is required; use --cpu-only for CPU Check")
    device = dml.device()
    check_factories(device)
    check_lifetimes(device)
    check_resampling(device)
    check_synchronization(device)
    if native_checks:
        client, module = native_checks
        client.tdml_check_synchronization.argtypes = [ctypes.c_void_p, ctypes.c_int]
        client.tdml_check_synchronization.restype = ctypes.c_int
        if client.tdml_check_synchronization(module._handle, device.index):
            raise AssertionError("Independent queue/fence check failed")
    else:
        print("Independent queue/fence check not run (use --native-checks)")
    print(f"PASS runtime checks on {dml.device_name(device.index)}")


if __name__ == "__main__":
    main()
