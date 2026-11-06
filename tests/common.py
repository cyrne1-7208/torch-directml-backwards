"""Shared numerical checks for the installed DirectML extension."""

import argparse

import torch

_checks = 0


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    if isinstance(value, dict):
        return {name: map_tensors(item, function) for name, item in value.items()}
    return value


def sample(shape, dtype=torch.float32):
    size = 1
    for dimension in shape:
        size *= dimension
    values = torch.arange(size, device="cpu").reshape(shape)
    if dtype is torch.bool:
        return values.remainder(3).ne(0)
    if dtype.is_floating_point:
        return (values.to(torch.float64).sin() * 0.7 + 0.2).to(dtype)
    return ((values * 5).remainder(13) - 6).to(dtype)


def output_like(value):
    fill = float("nan") if value.is_floating_point() else torch.iinfo(value.dtype).max
    return torch.full_like(value, fill)


def assert_result(
    actual, expected, device, *, dtype=None, rtol=None, atol=None, equal_nan=False
):
    global _checks
    if map_tensors(actual, lambda _: None) != map_tensors(expected, lambda _: None):
        raise AssertionError("The result structure or non-tensor values differ")
    actual_tensors = list(tensors(actual))
    expected_tensors = list(tensors(expected))
    if len(actual_tensors) != len(expected_tensors):
        raise AssertionError("The result contains a different number of tensors")
    for value, reference in zip(actual_tensors, expected_tensors):
        if value.device != device:
            raise AssertionError(f"Expected {device}, got {value.device}")
        wanted_dtype = (
            dtype
            if dtype is not None and reference.is_floating_point()
            else reference.dtype
        )
        if value.dtype != wanted_dtype:
            raise AssertionError(f"Expected {wanted_dtype}, got {value.dtype}")
        half = value.dtype is torch.float16
        tolerance = {
            "rtol": (5e-3 if half else 1e-4) if rtol is None else rtol,
            "atol": (5e-3 if half else 1e-5) if atol is None else atol,
        }
        # Integer width and index checks require exact equality, even for values above 2**24.
        if not value.is_floating_point() and not value.is_complex():
            tolerance = {"rtol": 0, "atol": 0}
        torch.testing.assert_close(
            value.detach().cpu(),
            reference.detach().cpu(),
            check_dtype=False,
            equal_nan=equal_nan,
            **tolerance,
        )
    _checks += 1


def check(
    label,
    function,
    args,
    device,
    *,
    kwargs=None,
    reference=None,
    gradients=False,
    inplace=False,
    rtol=None,
    atol=None,
    equal_nan=False,
):
    kwargs = {} if kwargs is None else kwargs

    def copy(value, target):
        return value.detach().to(target).clone().requires_grad_(value.requires_grad)

    cpu_args = map_tensors(args, lambda t: copy(t, "cpu"))
    cpu_kwargs = map_tensors(kwargs, lambda t: copy(t, "cpu"))
    gpu_args = map_tensors(args, lambda t: copy(t, device))
    gpu_kwargs = map_tensors(kwargs, lambda t: copy(t, device))
    try:
        expected = (reference or function)(*cpu_args, **cpu_kwargs)
        actual = function(*gpu_args, **gpu_kwargs)
        assert_result(
            actual, expected, device, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        # out= and in-place calls must return the supplied Tensor, not just an equal copy.
        output = (
            gpu_args[0]
            if inplace
            else gpu_kwargs.get("out", gpu_kwargs.get("grad_input"))
        )
        if output is not None and actual is not output:
            raise AssertionError("The result lost its input/output identity")
        if gradients:
            cpu_inputs = [t for t in tensors((cpu_args, cpu_kwargs)) if t.requires_grad]
            gpu_inputs = [t for t in tensors((gpu_args, gpu_kwargs)) if t.requires_grad]
            cpu_outputs = [t for t in tensors(expected) if t.requires_grad]
            gpu_outputs = [t for t in tensors(actual) if t.requires_grad]
            if (
                not cpu_inputs
                or not cpu_outputs
                or len(cpu_outputs) != len(gpu_outputs)
            ):
                raise AssertionError(
                    "The requested backward case has no differentiable path"
                )
            upstream = [
                torch.linspace(
                    -0.7, 0.9, t.numel(), dtype=t.dtype, device="cpu"
                ).reshape(t.shape)
                for t in cpu_outputs
            ]
            cpu_grad = torch.autograd.grad(cpu_outputs, cpu_inputs, upstream)
            gpu_grad = torch.autograd.grad(
                gpu_outputs, gpu_inputs, [t.to(device) for t in upstream]
            )
            assert_result(
                gpu_grad, cpu_grad, device, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        # Running statistics and in-place/out arguments are observable outputs too.
        assert_result(
            tuple(tensors((gpu_args, gpu_kwargs))),
            tuple(tensors((cpu_args, cpu_kwargs))),
            device,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    except Exception as error:
        raise AssertionError(f"{label}: {error}") from error
    print(f"PASS {label}", flush=True)


def run(function):
    parser = argparse.ArgumentParser(description=function.__doc__)
    parser.parse_args()
    import torch_directml as dml

    if not dml.is_available():
        raise RuntimeError("A DirectML GPU is required for numerical checks")
    device = dml.device()
    before = _checks
    function(device)
    if _checks == before:
        raise AssertionError("The selected group did not execute a numerical check")
    print(
        f"PASS {function.__name__}: {_checks - before} numerical checks on {dml.device_name(device.index)}"
    )
