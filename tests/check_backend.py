"""Check the installed Windows extension and DirectML GPU; use --cpu-only to skip GPU checks."""

import argparse
from math import prod
from pathlib import Path


def check_synchronization(device):
    from concurrent.futures import ThreadPoolExecutor

    import torch

    stream = torch.accelerator.current_stream(device)
    # Query and synchronize before readback; vary inputs and shapes to cover commands still being recorded.
    for size in (64, 96, 64):
        source = (
            torch.arange(size * size, dtype=torch.float32).sin().reshape(size, size)
        )
        value = source.to(device)
        result = value @ value.T
        stream.query()
        stream.synchronize()
        if not stream.query():
            raise AssertionError(
                "The default stream is incomplete after synchronization"
            )
        torch.testing.assert_close(
            result.cpu(), source @ source.T, rtol=1e-4, atol=1e-4
        )

        result = value + size
        torch.accelerator.synchronize(device)
        if not stream.query():
            raise AssertionError("Device synchronization did not complete pending work")
        torch.testing.assert_close(result.cpu(), source + size)

    def work(offset):
        source = torch.arange(256, dtype=torch.float32) + offset
        value = source.to(device)
        for _ in range(8):
            value = value + 1
            torch.accelerator.synchronize(device)
        return value.cpu(), source + 8

    # Concurrent submissions must not exclude work preceding each synchronization call.
    with ThreadPoolExecutor(max_workers=2) as pool:
        for actual, expected in pool.map(work, (1, 10)):
            torch.testing.assert_close(actual, expected)


def check_resampling(device):
    import torch
    from torch.nn import functional

    def check(source, **options):
        cpu = source.clone().requires_grad_()
        gpu = source.to(device).requires_grad_()
        expected = functional.interpolate(cpu, mode="bilinear", **options)
        actual = functional.interpolate(gpu, mode="bilinear", **options)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
        # Use nonuniform gradients to check boundary weights and backward coordinates.
        gradient = torch.linspace(-1, 1, expected.numel()).reshape(expected.shape)
        expected_gradient = torch.autograd.grad(expected, cpu, gradient)[0]
        actual_gradient = torch.autograd.grad(actual, gpu, gradient.to(device))[0]
        torch.testing.assert_close(
            actual_gradient.cpu(), expected_gradient, rtol=1e-4, atol=1e-5
        )

    # Change settings for the same shape to catch incorrect reuse of the first cached operator.
    for shape, size, order in (
        ((1, 2, 3, 4), (5, 7), (False, True, False)),
        ((1, 1, 5, 7), (3, 4), (True, False, True)),
        ((1, 1, 2, 3), (1, 1), (False, True)),
        ((1, 1, 1, 3), (3, 6), (True, False)),
    ):
        source = torch.arange(prod(shape), dtype=torch.float32).sin().reshape(shape)
        for align_corners in order:
            check(source, size=size, align_corners=align_corners)

    # Distinguish explicit scales from size-derived scales even when the output shapes match.
    source = torch.arange(15, dtype=torch.float32).sin().reshape(1, 1, 3, 5)
    for recompute in (False, True, False):
        check(
            source,
            scale_factor=1.5,
            align_corners=False,
            recompute_scale_factor=recompute,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run CPU checks without a DirectML GPU"
    )
    parser.add_argument(
        "--native-checks", type=Path, help="C++ checks DLL built with tests/native"
    )
    args = parser.parse_args()

    import torch
    from torch.nn import functional

    baseline = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    before = baseline + 1
    import torchvision

    import torch_directml as dml

    torch.testing.assert_close(baseline + 1, before)
    import torch_directml_native

    # Exercise argument-tuple ABI handling through the actual binding without initializing a GPU.
    if torch_directml_native.custom_device(0) != torch.device("privateuseone", 0):
        raise AssertionError("Native device binding returned a different device")
    try:
        torch_directml_native.custom_device("invalid")
    except TypeError:
        pass
    else:
        raise AssertionError("Native device binding accepted an invalid argument")
    for operator in ("aten::add.out", "aten::_copy_from"):
        if not torch._C._dispatch_has_kernel_for_dispatch_key(operator, "PrivateUse1"):
            raise AssertionError(f"DirectML kernel was not registered: {operator}")
    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept = torch.tensor([0, 2])
    torch.testing.assert_close(torchvision.ops.nms(boxes, scores, 0.5), kept)
    # The backend uses a CPU generator, allowing vtable and state checks without a GPU.
    generator = torch.Generator(device="privateuseone:0").manual_seed(123)
    if generator.initial_seed() != 123:
        raise AssertionError("Native generator did not retain its seed")
    torch.testing.assert_close(
        generator.clone_state().get_state(), generator.get_state()
    )
    if args.native_checks:
        import ctypes

        import lief

        code = lief.PE.parse(args.native_checks.with_name("native-code.bin"))
        symbols = {entry.name: entry.address for entry in code.get_export().entries}
        module = ctypes.WinDLL(torch_directml_native.__file__)
        marker = ctypes.c_char.in_dll(module, "tdml_torch_version")
        if ctypes.addressof(marker) - module._handle != symbols["tdml_torch_version"]:
            raise ValueError(
                "Native checks require the code used to build the installed extension"
            )
        client = ctypes.CDLL(str(args.native_checks.resolve()))
        client.tdml_check_native.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        client.tdml_check_native.restype = ctypes.c_int
        if (
            client.tdml_check_native(
                module._handle, symbols["tdml_nms"], symbols["tdml_tensor_construct"]
            )
            != 0
        ):
            raise AssertionError("In-module native checks failed")
    print(
        f"PASS: import, operator registration, and generator on PyTorch {torch.__version__}"
    )
    if args.cpu_only:
        print("GPU checks skipped (--cpu-only)")
        return

    if not dml.is_available():
        raise RuntimeError("A DirectML GPU is required; use --cpu-only for CPU Check")
    device = dml.device()
    check_synchronization(device)
    value = baseline.to(device)
    torch.testing.assert_close(value.cpu(), baseline)
    torch.testing.assert_close(value.detach().cpu(), baseline)
    torch.testing.assert_close(value.T.detach().cpu(), baseline.T)
    torch.testing.assert_close(value.sum(1).detach().cpu(), baseline.sum(1))
    torch.testing.assert_close((value + 1).cpu(), before)
    torch.testing.assert_close(value.T.contiguous().cpu(), baseline.T.contiguous())
    torch.testing.assert_close((value @ value.T).cpu(), baseline @ baseline.T)
    reciprocal_input = (value + 1).detach().requires_grad_()
    reciprocal = reciprocal_input.reciprocal()
    torch.testing.assert_close(reciprocal.cpu(), (baseline + 1).reciprocal())
    reciprocal.sum().backward()
    torch.testing.assert_close(
        reciprocal_input.grad.cpu(), -(baseline + 1).reciprocal().square()
    )
    torch.testing.assert_close(
        torchvision.ops.nms(boxes.to(device), scores.to(device), 0.5).cpu(), kept
    )

    # Exercise string-argument ABI handling and Tensor lifetimes through div, gelu and backward.
    torch.testing.assert_close(
        torch.div(value + 1, 2, rounding_mode="floor").cpu(),
        torch.div(baseline + 1, 2, rounding_mode="floor"),
    )
    torch.testing.assert_close(
        functional.gelu(value, approximate="tanh").cpu(),
        functional.gelu(baseline, approximate="tanh"),
        rtol=1e-4,
        atol=1e-5,
    )
    differentiable = value.detach().requires_grad_()
    (differentiable * differentiable).sum().backward()
    torch.testing.assert_close(differentiable.grad.cpu(), baseline * 2)
    check_resampling(device)

    # Check Tensor arguments in the custom API separately from ordinary torch operators.
    weight = torch.ones(6, device=device)
    expected = baseline * torch.rsqrt(baseline.square().mean(-1, keepdim=True) + 1e-5)
    torch.testing.assert_close(
        dml.rmsnorm(value, weight, 1e-5).cpu(), expected, rtol=1e-4, atol=1e-5
    )
    # MHA and RoPE construct tuples separately; check both return paths.
    attention_expected = torch.ones(1, 2, 8)
    attention_input = attention_expected.to(device)
    output, past_key, past_value = dml.multi_head_attention(
        attention_input, attention_input, attention_input, 8, 2, None, None
    )
    torch.testing.assert_close(output.cpu(), attention_expected)
    for cache in (past_key, past_value):
        torch.testing.assert_close(cache.cpu(), torch.ones(1, 2, 2, 4))

    rotary_expected = torch.ones(1, 1, 2, 4)
    rotary_input = rotary_expected.to(device)
    rotated_query, rotated_key = dml.apply_rotary_position_emb(
        rotary_input,
        rotary_input,
        torch.ones_like(rotary_input),
        torch.zeros_like(rotary_input),
        0,
        2,
        4,
    )
    for rotated in (rotated_query, rotated_key):
        torch.testing.assert_close(rotated.cpu(), rotary_expected)
    print(
        f"PASS: transfers, views, operators, autograd, and native binding on {dml.device_name(0)}"
    )


if __name__ == "__main__":
    main()
