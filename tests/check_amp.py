"""Check installed Windows FP16 AMP; use --cpu-only to skip DirectML execution."""

import argparse
import threading

DEVICE_TYPE = "privateuseone"


def check_condition(condition, message):
    if not condition:
        raise AssertionError(message)


def check_close(torch, actual, expected, label, rtol=1e-2, atol=1e-2):
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        raise AssertionError(f"{label}: {error}") from error


def has_kernel(torch, name, key):
    return torch._C._dispatch_has_kernel_for_dispatch_key(name, key)


def step_with_scaler(scaler, optimizer):
    scaler.step(optimizer)
    scaler.update()


def check_autocast_values(torch, dml, enabled, dtype, label):
    module = dml.PrivateUse1Module
    reported_state = module.is_autocast_enabled(), module.get_autocast_dtype()
    canonical_state = (
        torch.is_autocast_enabled(DEVICE_TYPE),
        torch.get_autocast_dtype(DEVICE_TYPE),
    )
    check_condition(
        reported_state == canonical_state == (enabled, dtype),
        f"{label}: {reported_state}/{canonical_state}",
    )


def check_autocast_state(torch, dml):
    module = dml.PrivateUse1Module
    original_state = module.is_autocast_enabled(), module.get_autocast_dtype()
    check_autocast_values(torch, dml, *original_state, "initial")
    try:
        module.set_autocast_enabled(False)
        module.set_autocast_dtype(torch.float32)
        check_autocast_values(torch, dml, False, torch.float32, "forwarded setters")
        with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
            check_autocast_values(torch, dml, True, torch.float16, "enter")
            with torch.autocast(DEVICE_TYPE, dtype=torch.float16, enabled=False):
                check_autocast_values(
                    torch, dml, False, torch.float16, "nested disable"
                )
            check_autocast_values(torch, dml, True, torch.float16, "nested restore")
            thread_states = []
            worker = threading.Thread(
                target=lambda: thread_states.append(module.is_autocast_enabled())
            )
            worker.start()
            worker.join()
            check_condition(
                thread_states == [False], f"thread inherited AMP: {thread_states}"
            )
        check_autocast_values(torch, dml, False, torch.float32, "context restore")

        class ProbeError(Exception):
            pass

        try:
            with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
                raise ProbeError
        except ProbeError:
            pass
        check_autocast_values(torch, dml, False, torch.float32, "exception restore")
    finally:
        module.set_autocast_enabled(original_state[0])
        module.set_autocast_dtype(original_state[1])


def check_registrations(torch):
    check_condition(
        torch.privateuseone.get_amp_supported_dtype() == [torch.float16]
        and torch.amp.autocast_mode.is_autocast_available(DEVICE_TYPE),
        "FP16 autocast unavailable",
    )
    for operator in [
        "mm",
        "matmul",
        "bmm",
        "addmm",
        "baddbmm",
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "convolution",
    ]:
        check_condition(
            has_kernel(torch, f"aten::{operator}", "AutocastPrivateUse1"),
            f"missing policy: {operator}",
        )
    for operator in (
        "_amp_foreach_non_finite_check_and_unscale_",
        "_amp_update_scale_",
    ):
        check_condition(
            has_kernel(torch, f"aten::{operator}", "PrivateUse1"),
            f"missing scaler helper: {operator}",
        )


def check_layers(torch, device):
    from torch import nn

    reference_models = nn.ModuleList(
        [nn.Linear(4, 3), nn.Conv2d(2, 3, 3, padding=1), nn.BatchNorm2d(3)]
    )
    models = nn.ModuleList(
        [nn.Linear(4, 3), nn.Conv2d(2, 3, 3, padding=1), nn.BatchNorm2d(3)]
    )
    models.load_state_dict(reference_models.state_dict())
    models.to(device)
    linear_reference, convolution_reference, batch_norm_reference = reference_models
    linear, convolution, batch_norm = models
    parameters = [
        *linear.parameters(),
        *convolution.parameters(),
        *batch_norm.parameters(),
    ]
    parameter_dtypes = [parameter.dtype for parameter in parameters]
    running_mean_before = batch_norm.running_mean.detach().cpu().clone()
    inputs = torch.linspace(-0.8, 0.9, 12).reshape(3, 4)
    image = torch.linspace(-0.8, 0.9, 50).reshape(1, 2, 5, 5)
    with torch.no_grad(), torch.autocast(DEVICE_TYPE, dtype=torch.float16):
        output = linear(inputs.to(device))
        convolution_output = convolution(image.to(device))
        normalized_output = batch_norm(convolution_output)
    convolution_reference_output = convolution_reference(image)
    normalized_reference = batch_norm_reference(convolution_reference_output)
    check_condition(
        output.dtype
        is convolution_output.dtype
        is normalized_output.dtype
        is torch.float16,
        "AMP layer output dtype",
    )
    check_close(torch, output.float().cpu(), linear_reference(inputs), "Linear")
    check_close(
        torch, convolution_output.float().cpu(), convolution_reference_output, "Conv2d"
    )
    check_close(
        torch, normalized_output.float().cpu(), normalized_reference, "BatchNorm2d"
    )
    check_condition(
        all(parameter.dtype is torch.float32 for parameter in parameters)
        and parameter_dtypes == [parameter.dtype for parameter in parameters],
        "autocast changed FP32 parameters",
    )
    check_condition(
        batch_norm.running_mean.dtype is batch_norm.running_var.dtype is torch.float32
        and not torch.equal(batch_norm.running_mean.cpu(), running_mean_before),
        "BatchNorm stats dtype/update",
    )
    check_close(
        torch,
        batch_norm.running_mean.cpu(),
        batch_norm_reference.running_mean,
        "BatchNorm running_mean",
    )
    check_close(
        torch,
        batch_norm.running_var.cpu(),
        batch_norm_reference.running_var,
        "BatchNorm running_var",
    )


def check_fp32_loss(torch, device):
    from torch.nn import functional

    inputs = torch.tensor([[1.0001, -0.9999, 0.9998, -1.0002]])
    target = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
        loss = functional.mse_loss(inputs.to(device), target.to(device))
    check_condition(
        loss.dtype is torch.float32 and loss.item() > 0, "sensitive MSE was not FP32"
    )
    check_close(
        torch,
        loss.cpu(),
        functional.mse_loss(inputs, target),
        "FP32 MSE",
        rtol=1e-3,
        atol=1e-10,
    )


def check_exclusions(torch, device):
    from torch.nn import functional

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    b = torch.tensor([[0.5, -1.0], [1.5, 0.25]], device=device)
    bias = torch.zeros((2, 2), device=device)
    with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
        autocast_result = torch.addmm(bias, a, b)
        out_result = torch.empty_like(bias)
        torch.addmm(bias, a, b, out=out_result)
        inplace_result = bias.clone().addmm_(a, b)
        explicit_dtype_output = functional.softmax(
            torch.tensor([[1.0, 2.0]], device=device), -1, dtype=torch.float16
        )
    check_condition(autocast_result.dtype is torch.float16, "addmm was not autocast")
    check_condition(
        out_result.dtype is inplace_result.dtype is torch.float32,
        "out= or in-place addmm was autocast",
    )
    check_condition(
        explicit_dtype_output.dtype is torch.float16, "softmax ignored explicit dtype"
    )
    check_close(
        torch,
        autocast_result.float().cpu(),
        torch.addmm(bias.cpu(), a.cpu(), b.cpu()),
        "addmm",
    )
    check_condition(
        torch.equal(out_result.cpu(), inplace_result.cpu()),
        "out= and in-place addmm differ",
    )
    try:
        with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
            functional.binary_cross_entropy(
                torch.full((2,), 0.5, device=device), torch.ones(2, device=device)
            )
    except RuntimeError as error:
        message = str(error).lower()
        check_condition(
            "unsafe" in message and "autocast" in message,
            f"unexpected BCE error: {error}",
        )
    else:
        raise AssertionError("binary_cross_entropy was accepted under autocast")


def check_scaler_training(torch, device):
    from torch.nn import functional

    reference_model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        reference_model.weight.copy_(torch.tensor([[0.3, -0.2]]))
    model = torch.nn.Linear(2, 1, bias=False)
    model.load_state_dict(reference_model.state_dict())
    model.to(device)
    batches = (
        (torch.tensor([[-1.0, 0.5], [0.25, 1.5]]), torch.tensor([[0.2], [-0.1]])),
        (torch.tensor([[1.0, -0.5], [0.75, 0.25]]), torch.tensor([[0.5], [0.3]])),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(DEVICE_TYPE, init_scale=128, growth_interval=1)
    optimizer.zero_grad(set_to_none=True)
    for inputs, target in batches:
        reference_output = reference_model(inputs)
        (functional.mse_loss(reference_output, target) / len(batches)).backward()
        with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
            output = model(inputs.to(device))
            loss = functional.mse_loss(output, target.to(device)) / len(batches)
        check_condition(
            output.dtype is torch.float16 and loss.dtype is torch.float32,
            "training dtype",
        )
        check_close(
            torch, output.float().cpu(), reference_output.detach(), "training output"
        )
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    # Compare against CPU FP32 gradients with tolerance for FP16 backward rounding.
    check_close(
        torch, model.weight.grad.cpu(), reference_model.weight.grad, "grad", 2e-2, 2e-3
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 0.05, foreach=False
    )
    check_condition(
        gradient_norm.item() > 0.05
        and torch.linalg.vector_norm(model.weight.grad).item() <= 0.0505,
        "gradient clipping was not exercised",
    )
    weights_before = model.weight.detach().cpu().clone()
    step_with_scaler(scaler, optimizer)
    check_condition(
        not torch.equal(model.weight.cpu(), weights_before),
        "finite step did not update",
    )
    check_condition(scaler.get_scale() == 256, "finite step did not grow scale")

    inputs, target = batches[0]
    for label, factor in (("Inf", float("inf")), ("NaN", float("nan"))):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
            bad_loss = model(inputs.to(device)).float().sum() * factor
        scaler.scale(bad_loss).backward()
        check_condition(
            any(
                not torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            ),
            f"{label} did not produce non-finite gradients",
        )
        weights_before, previous_scale = model.weight.cpu().clone(), scaler.get_scale()
        step_with_scaler(scaler, optimizer)
        check_condition(
            torch.equal(model.weight.cpu(), weights_before),
            f"{label} update was not skipped",
        )
        check_condition(
            scaler.get_scale() == previous_scale * 0.5,
            f"{label} did not back off scale",
        )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(DEVICE_TYPE, dtype=torch.float16):
        loss = functional.mse_loss(model(inputs.to(device)), target.to(device))
    scaler.scale(loss).backward()
    weights_before = model.weight.cpu().clone()
    step_with_scaler(scaler, optimizer)
    check_condition(
        not torch.equal(model.weight.cpu(), weights_before),
        "recovery step did not update",
    )
    check_condition(scaler.get_scale() == 128, "recovery did not regrow scale")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only", action="store_true", help="Run CPU checks without a DirectML GPU"
    )
    args = parser.parse_args()
    import torch

    import torch_directml as dml

    check_autocast_state(torch, dml)
    print("PASS autocast forwarding, nesting, exceptions, and thread isolation")
    check_registrations(torch)
    print("PASS AMP registrations")
    if args.cpu_only:
        print("GPU checks skipped (--cpu-only)")
        return
    if not dml.is_available():
        raise RuntimeError("A DirectML GPU is required; use --cpu-only for CPU Check")
    device = dml.device()
    check_condition(device.type == DEVICE_TYPE, f"unexpected device type: {device}")
    check_layers(torch, device)
    print("PASS Linear, Conv2d, and BatchNorm2d autocast/training state")
    check_fp32_loss(torch, device)
    check_exclusions(torch, device)
    check_scaler_training(torch, device)
    print(f"PASS FP16 SGD/GradScaler on {dml.device_name(0)}")


if __name__ == "__main__":
    main()
