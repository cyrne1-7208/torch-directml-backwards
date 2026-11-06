"""Compare matrix and neural-network operators with CPU references."""

from functools import partial

import torch
from common import assert_result, check, map_tensors, run, sample
from torch.nn import functional


def matrix_cases():
    """Share input geometry between ordinary execution and the FP16 autocast policy checks."""

    def value(shape):
        return sample(shape).requires_grad_()

    yield "mm", torch.mm, (value((2, 3)), value((3, 4))), {}
    yield "matmul", torch.matmul, (value((2, 1, 3, 4)), value((1, 5, 4, 2))), {}
    yield "bmm", torch.bmm, (value((2, 3, 4)), value((2, 4, 2))), {}
    yield (
        "addmm",
        torch.addmm,
        (value((1, 3)), value((2, 4)), value((4, 3))),
        {"alpha": 1.2, "beta": 0.7},
    )
    yield (
        "baddbmm",
        torch.baddbmm,
        (value((1, 2, 2)), value((2, 2, 3)), value((2, 3, 2))),
        {"alpha": 1.3, "beta": 0.6},
    )
    yield (
        "linear",
        functional.linear,
        (value((2, 2, 4)), value((3, 4)), value((3,))),
        {},
    )
    # The pinned backend requires 4D convolution inputs; ATen expands 1D inputs to 2D.
    for dimensions in (1, 2):
        shape = (1, 4, *((5,) * dimensions))
        weights = (4, 2, *((3,) * dimensions))
        options = {"stride": 2, "padding": 1, "dilation": 2, "groups": 2}
        yield (
            f"conv{dimensions}d",
            getattr(functional, f"conv{dimensions}d"),
            (value(shape), value(weights), value((4,))),
            options,
        )
        shape = (1, 4, *((3,) * dimensions))
        options = {"stride": 2, "padding": 1, "output_padding": 1, "groups": 2}
        yield (
            f"conv_transpose{dimensions}d",
            getattr(functional, f"conv_transpose{dimensions}d"),
            (value(shape), value(weights), value((4,))),
            options,
        )
    yield (
        "convolution",
        torch.ops.aten.convolution.default,
        (
            value((1, 2, 5, 5)),
            value((3, 2, 3, 3)),
            value((3,)),
            [1, 2],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
        ),
        {},
    )


def check_training_chain(device):
    reference_model = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)
    )
    with torch.no_grad():
        reference_model[0].weight.copy_(torch.linspace(-0.3, 0.3, 12).reshape(3, 4))
        reference_model[0].bias.copy_(torch.tensor([-0.1, 0.08, 0.03]))
        reference_model[2].weight.copy_(torch.linspace(-0.25, 0.35, 6).reshape(2, 3))
        reference_model[2].bias.copy_(torch.tensor([0.04, -0.07]))

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)
    )
    model.load_state_dict(reference_model.state_dict())
    model.to(device)

    inputs = torch.tensor([[-0.8, 0.2, 0.5, 1.0], [0.4, -0.6, 0.9, -0.1]])
    targets = torch.tensor([[0.3, -0.2], [-0.4, 0.7]])
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step, (batch_inputs, batch_targets) in enumerate(
        ((inputs, targets), (inputs.flip(0) + 0.15, targets.flip(0) - 0.1))
    ):
        reference_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)

        expected_output = reference_model(batch_inputs)
        actual_output = model(batch_inputs.to(device))
        expected_loss = functional.mse_loss(expected_output, batch_targets)
        actual_loss = functional.mse_loss(actual_output, batch_targets.to(device))
        assert_result(actual_output, expected_output, device)
        assert_result(actual_loss, expected_loss, device)

        expected_loss.backward()
        actual_loss.backward()
        for actual_parameter, expected_parameter in zip(
            model.parameters(), reference_model.parameters()
        ):
            assert_result(actual_parameter.grad, expected_parameter.grad, device)

        optimizer.step()
        reference_optimizer.step()
        for actual_parameter, expected_parameter in zip(
            model.parameters(), reference_model.parameters()
        ):
            assert_result(actual_parameter, expected_parameter, device)
        print(f"PASS training-chain/step={step}", flush=True)


def half_reference(function, *args, **kwargs):
    # Evaluate FP16-rounded inputs in FP32; autograd rounds the output and leaf gradients back to FP16.
    return function(*map_tensors(args, lambda value: value.float()), **kwargs).half()


def check_nn(device):
    """Check forward and backward results for matrix and neural-network operators."""
    for name, function, args, options in matrix_cases():
        check(name, function, args, device, kwargs=options, gradients=True)
        half_args = map_tensors(args, lambda value: value.half())

        check(
            f"{name}/float16",
            function,
            half_args,
            device,
            kwargs=options,
            reference=partial(half_reference, function),
            gradients=True,
        )

    pool_input = sample((1, 2, 4, 5)).requires_grad_()
    check(
        "avg_pool2d padded",
        functional.avg_pool2d,
        (pool_input,),
        device,
        kwargs={
            "kernel_size": (2, 3),
            "stride": (1, 2),
            "padding": (0, 1),
            "count_include_pad": False,
        },
        gradients=True,
    )

    max_pool_input = sample((1, 2, 4, 5)).requires_grad_()
    check(
        "max_pool2d indices",
        functional.max_pool2d,
        (max_pool_input,),
        device,
        kwargs={"kernel_size": 2, "stride": 1, "return_indices": True},
        gradients=True,
    )

    adaptive_input = sample((1, 2, 4, 5)).requires_grad_()
    check(
        "adaptive_avg_pool2d",
        functional.adaptive_avg_pool2d,
        (adaptive_input, (2, 3)),
        device,
        gradients=True,
    )

    adaptive_max_input = sample((1, 2, 4, 5)).requires_grad_()
    check(
        "adaptive_max_pool2d indices",
        functional.adaptive_max_pool2d,
        (adaptive_max_input, (2, 3)),
        device,
        kwargs={"return_indices": True},
        gradients=True,
    )

    pad_input = sample((1, 2, 3, 4)).requires_grad_()
    check(
        "constant padding",
        functional.pad,
        (pad_input, (1, 2, 0, 1)),
        device,
        kwargs={"mode": "constant", "value": 0.25},
        gradients=True,
    )
    reflect_input = sample((1, 2, 3, 4)).requires_grad_()
    check(
        "reflect padding",
        functional.pad,
        (reflect_input, (1, 1, 1, 1)),
        device,
        kwargs={"mode": "reflect"},
        gradients=True,
    )
    check(
        "replicate padding",
        functional.pad,
        (pad_input,),
        device,
        kwargs={"pad": (1, 2, 0, 1), "mode": "replicate"},
        gradients=True,
    )

    bn_input = (
        sample((2, 3, 2, 2)) * 0.01
        + torch.tensor([0.0, 0.1, -0.08]).reshape(1, 3, 1, 1)
    ).requires_grad_()
    bn_weight = torch.tensor([0.65, 1.25, 0.8], requires_grad=True)
    bn_bias = torch.tensor([-0.2, 0.15, 0.35], requires_grad=True)
    bn_mean = torch.tensor([0.1, -0.2, 0.15])
    bn_var = torch.tensor([0.7, 1.4, 0.5])
    check(
        "batch_norm training",
        functional.batch_norm,
        (bn_input, bn_mean, bn_var, bn_weight, bn_bias),
        device,
        kwargs={"training": True, "momentum": 0.3, "eps": 1e-3},
        gradients=True,
    )

    bn_eval_input = bn_input.detach().clone().requires_grad_()
    bn_eval_mean = torch.tensor([0.1, -0.2, 0.15])
    bn_eval_var = torch.tensor([0.7, 1.4, 0.5])
    bn_eval_weight = bn_weight.detach().clone().requires_grad_()
    bn_eval_bias = bn_bias.detach().clone().requires_grad_()
    check(
        "batch_norm eval",
        functional.batch_norm,
        (bn_eval_input, bn_eval_mean, bn_eval_var, bn_eval_weight, bn_eval_bias),
        device,
        kwargs={"training": False, "momentum": 0.3, "eps": 1e-3},
        gradients=True,
    )

    layer_input = (sample((2, 3, 4)) * 0.01).requires_grad_()
    layer_weight = torch.linspace(0.6, 1.4, 12).reshape(3, 4).requires_grad_()
    layer_bias = torch.linspace(-0.2, 0.3, 12).reshape(3, 4).requires_grad_()
    check(
        "layer_norm",
        functional.layer_norm,
        (layer_input, (3, 4), layer_weight, layer_bias),
        device,
        kwargs={"eps": 2e-3},
        gradients=True,
    )

    group_input = (sample((2, 4, 2, 3)) * 0.01).requires_grad_()
    group_weight = torch.tensor([0.7, 1.2, 0.85, 1.35], requires_grad=True)
    group_bias = torch.tensor([-0.15, 0.2, 0.05, -0.3], requires_grad=True)
    check(
        "group_norm",
        functional.group_norm,
        (group_input, 2, group_weight, group_bias),
        device,
        kwargs={"eps": 2e-3},
        gradients=True,
    )

    prediction = torch.tensor([[-0.6, 0.2, 0.9], [0.4, -0.3, 0.7]], requires_grad=True)
    target = torch.tensor([[0.1, -0.2, 0.5], [-0.5, 0.6, 0.2]])
    check("mse_loss", functional.mse_loss, (prediction, target), device, gradients=True)
    check("l1_loss", functional.l1_loss, (prediction, target), device, gradients=True)
    check(
        "smooth_l1_loss",
        functional.smooth_l1_loss,
        (prediction, target),
        device,
        kwargs={"beta": 0.35},
        gradients=True,
    )

    logits = torch.tensor([[-1.2, 0.4, 1.1], [0.8, -0.7, 0.2]], requires_grad=True)
    binary_target = torch.tensor([[0.0, 0.8, 1.0], [0.3, 0.0, 0.65]])
    check(
        "binary_cross_entropy_with_logits",
        functional.binary_cross_entropy_with_logits,
        (logits, binary_target),
        device,
        gradients=True,
    )
    check(
        "binary_cross_entropy/outside-autocast",
        functional.binary_cross_entropy,
        (
            torch.tensor([[0.2, 0.7, 0.9], [0.6, 0.3, 0.4]], requires_grad=True),
            binary_target,
        ),
        device,
        gradients=True,
    )
    log_probabilities = functional.log_softmax(sample((3, 4)), dim=1).requires_grad_()
    check(
        "nll_loss/weighted-ignore-index",
        functional.nll_loss,
        (log_probabilities, torch.tensor([1, 3, -100])),
        device,
        kwargs={"weight": torch.tensor([0.6, 1.2, 0.9, 1.4]), "ignore_index": -100},
        gradients=True,
    )
    check(
        "nll_loss2d/weighted-ignore-index",
        functional.nll_loss,
        (
            functional.log_softmax(sample((1, 4, 2, 3)), dim=1).requires_grad_(),
            torch.tensor([[[1, 3, -100], [0, 2, 1]]]),
        ),
        device,
        kwargs={"weight": torch.tensor([0.6, 1.2, 0.9, 1.4]), "ignore_index": -100},
        gradients=True,
    )

    class_logits = torch.tensor(
        [[-0.3, 0.7, 0.2, -0.1], [0.5, -0.4, 0.1, 0.9]], requires_grad=True
    )
    class_target = torch.tensor([1, 3], dtype=torch.int64)
    check(
        "cross_entropy",
        functional.cross_entropy,
        (class_logits, class_target),
        device,
        gradients=True,
    )
    check(
        "prelu/per-channel-weight",
        functional.prelu,
        (
            torch.tensor(
                [[[-0.8, 0.2, -0.3, 0.9], [0.4, -0.7, 0.6, -0.1]]], requires_grad=True
            ),
            torch.tensor([0.2, 0.7], requires_grad=True),
        ),
        device,
        gradients=True,
    )
    check(
        "cdist/batched",
        torch.cdist,
        (
            sample((2, 3, 4)).requires_grad_(),
            (sample((2, 2, 4)).flip(-1) * 0.7 + 0.4).requires_grad_(),
        ),
        device,
        gradients=True,
    )
    check(
        "dot",
        torch.dot,
        (sample((7,)).requires_grad_(), sample((7,)).flip(0).requires_grad_()),
        device,
        gradients=True,
    )
    for probability, training in ((0.0, True), (1.0, True), (0.25, False)):
        check(
            f"dropout/p={probability}/training={training}",
            functional.dropout,
            (sample((2, 3)).requires_grad_(),),
            device,
            kwargs={"p": probability, "training": training},
            gradients=True,
        )

    value = torch.ones((17, 11), device=device, requires_grad=True)
    dropped = functional.dropout(value, p=0.25, training=True)
    if not ((dropped == 0) | (dropped == 4 / 3)).all().item():
        raise AssertionError("Dropout returned an invalid masked/scaled value")
    if not (dropped == 0).any().item() or not (dropped != 0).any().item():
        raise AssertionError(
            "Dropout backward case must contain both kept and dropped entries"
        )
    upstream = sample((17, 11))
    (gradient,) = torch.autograd.grad(dropped, value, upstream.to(device))
    assert_result(gradient, dropped.detach().cpu() * upstream, device)

    weights = [
        sample(shape).requires_grad_() for shape in ((12, 3), (12, 4), (12,), (12,))
    ]
    check(
        "gru.input/forward-and-backward",
        torch.ops.aten.gru.input,
        (
            sample((2, 3, 3)).requires_grad_(),
            sample((1, 2, 4)).requires_grad_(),
            weights,
            True,
            1,
            0.0,
            True,
            False,
            True,
        ),
        device,
        gradients=True,
    )
    from torchvision.ops import roi_align

    for aligned in (False, True):
        check(
            f"roi_align/aligned={aligned}/forward-and-backward",
            roi_align,
            (
                sample((2, 3, 6, 7)).requires_grad_(),
                torch.tensor([[0, 0.2, 0.3, 4.6, 4.2], [1, 1.1, 0.5, 5.2, 4.8]]),
                (2, 3),
            ),
            device,
            kwargs={"sampling_ratio": 2, "aligned": aligned},
            gradients=True,
        )
    check_training_chain(device)


if __name__ == "__main__":
    run(check_nn)
