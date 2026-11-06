"""Check pointwise, reduction, indexing and layout operations against CPU."""

import torch
from common import assert_result, check, output_like, run, sample
from torch.nn import functional


def check_ops(device):
    """Check supported numerical paths, including dtype and overload boundaries."""
    floats = (torch.float16, torch.float32)
    integers = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    for dtype in (*floats, *integers):
        x = sample((2, 3), dtype)
        y = torch.tensor([[1, 2, 3]], dtype=dtype)
        for name in ("add", "sub", "mul", "minimum", "maximum"):
            check(f"{name}/{dtype}/broadcast", getattr(torch, name), (x, y), device)
        for name in ("eq", "ne", "lt", "le", "gt", "ge"):
            check(f"{name}/{dtype}", getattr(torch, name), (x, y), device)
        check(f"where/{dtype}", torch.where, (mask, x, -x), device)
        for name in ("sum", "prod", "amin", "amax", "argmin", "argmax"):
            check(
                f"{name}/{dtype}/negative-axis",
                getattr(torch, name),
                (x,),
                device,
                kwargs={"dim": -1, "keepdim": True},
            )
        check(f"cumsum/{dtype}", torch.cumsum, (x, 1), device)
        clip_input = torch.tensor([[-3, 0, 2], [3, 5, 7]]).to(dtype)
        for bounds in ({"min": 1, "max": 4}, {"min": 1}, {"max": 4}):
            check(
                f"clamp/{dtype}/{bounds}",
                torch.clamp,
                (clip_input,),
                device,
                kwargs=bounds,
            )
        check(
            f"add.out/{dtype}",
            torch.add,
            (x, y),
            device,
            kwargs={"out": output_like(x)},
        )
        check(f"mul_/{dtype}", lambda a: a.mul_(2), (x,), device, inplace=True)

    boolean = sample((2, 3), torch.bool)
    check("where/bool", torch.where, (mask, boolean, ~boolean), device)
    # Values beyond FP32's exact integer range expose accidental float intermediates.
    for dtype, magnitude in ((torch.int32, 2**25), (torch.int64, 2**45)):
        x = torch.tensor([-magnitude - 3, -1, 0, magnitude + 5], dtype=dtype)
        for name in ("add", "sub", "eq", "lt"):
            check(f"{name}/{dtype}/wide-values", getattr(torch, name), (x, 1), device)
        check(f"copy/{dtype}/wide-values", lambda a: a.clone(), (x,), device)

    for dtype in (*integers, torch.bool):
        x = torch.tensor([[0, 0, 0], [1, 3, 5], [0, 1, 2]], dtype=dtype)
        y = torch.tensor([[1, 0, 6]], dtype=dtype)
        for name in (
            "bitwise_and",
            "bitwise_or",
            "bitwise_xor",
            "logical_and",
            "logical_or",
            "logical_xor",
        ):
            check(f"{name}/{dtype}", getattr(torch, name), (x, y), device)
        check(f"logical_not/{dtype}", torch.logical_not, (x,), device)
        check(f"bitwise_not/{dtype}", torch.bitwise_not, (x,), device)
        check(f"any/{dtype}", torch.any, (x,), device, kwargs={"dim": -1})
        check(f"all/{dtype}", torch.all, (x,), device, kwargs={"dim": -1})

    unary = (
        "abs",
        "neg",
        "sign",
        "square",
        "ceil",
        "floor",
        "round",
        "trunc",
        "frac",
        "reciprocal",
        "rsqrt",
        "sqrt",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "acosh",
        "atan",
        "asinh",
        "atanh",
        "exp2",
        "sinc",
        "sinh",
        "cosh",
        "tanh",
        "erf",
        "erfc",
        "sigmoid",
    )
    for dtype in floats:
        # This domain is valid for logs, roots and inverse trigonometric functions.
        x = torch.tensor([[0.125, 0.3, 0.7], [0.9, 0.4, 0.6]], dtype=dtype)
        for name in unary:
            values = (
                x
                if name in {"rsqrt", "sqrt", "log", "log1p", "log2", "log10"}
                else x - 0.5
            )
            if name in {"ceil", "floor", "round", "trunc", "frac"}:
                values = torch.tensor(
                    [[-2.7, -1.5, -0.3], [0.125, 1.5, 2.7]], dtype=dtype
                )
            if name == "acosh":
                values = x + 1.1
            if name == "sinc":
                values = torch.tensor([[-0.5, 0.0, 0.5], [-1.0, 1.0, 2.0]], dtype=dtype)
            check(f"{name}/{dtype}", getattr(torch, name), (values,), device)
        for name in (
            "true_divide",
            "floor_divide",
            "fmod",
            "remainder",
            "atan2",
            "pow",
        ):
            check(
                f"{name}/{dtype}", getattr(torch, name), (x, x.flip(-1) + 0.2), device
            )
        for name in ("mean", "var", "std", "logsumexp"):
            check(
                f"{name}/{dtype}/axis",
                getattr(torch, name),
                (x,),
                device,
                kwargs={"dim": 1, "keepdim": True},
            )
        check(
            f"sum/{dtype}/explicit-dtype",
            torch.sum,
            (x,),
            device,
            kwargs={"dim": (0, 1), "dtype": torch.float32},
        )
        check(f"softmax/{dtype}", functional.softmax, (x, -1), device)
        check(f"log_softmax/{dtype}", functional.log_softmax, (x, 0), device)
        for name in ("addcmul", "addcdiv"):
            check(
                f"{name}/{dtype}",
                getattr(torch, name),
                (x - 0.5, x, x.flip(-1)),
                device,
                kwargs={"value": 0.7},
            )
        for mode in (None, "floor", "trunc"):
            # Explicit Tensor/Tensor out_mode reaches the adapted optional-string schema.
            check(
                f"div.out_mode/{dtype}/{mode}",
                torch.div,
                (-x * 7, x.flip(-1)),
                device,
                kwargs={"rounding_mode": mode, "out": output_like(x)},
            )
        gelu_input = torch.tensor([[-2.5, -1.2, -0.2], [0.5, 1.3, 2.2]], dtype=dtype)
        for approximate in ("none", "tanh"):
            check(
                f"gelu.out/{dtype}/{approximate}",
                torch.ops.aten.gelu.out,
                (gelu_input,),
                device,
                kwargs={"approximate": approximate, "out": output_like(gelu_input)},
            )
            check(
                f"gelu_backward.grad_input/{dtype}/{approximate}",
                torch.ops.aten.gelu_backward.grad_input,
                (x.flip(-1), gelu_input),
                device,
                kwargs={
                    "approximate": approximate,
                    "grad_input": output_like(gelu_input),
                },
            )

    for name in ("sin", "exp", "log", "sqrt", "tanh", "sigmoid", "reciprocal"):
        x = torch.linspace(0.2, 0.8, 12).reshape(3, 4).requires_grad_()
        check(f"{name}/backward", getattr(torch, name), (x,), device, gradients=True)

    for name in ("sum", "prod", "mean", "var", "std", "logsumexp", "cumsum"):
        check(
            f"{name}/backward",
            getattr(torch, name),
            (sample((2, 3)).requires_grad_(),),
            device,
            kwargs={"dim": 1},
            gradients=True,
        )
    check(
        "vector_norm/backward",
        torch.linalg.vector_norm,
        (sample((2, 3)).requires_grad_(),),
        device,
        kwargs={"dim": 1},
        gradients=True,
    )

    for name, options in (
        ("relu", {}),
        ("leaky_relu", {"negative_slope": 0.2}),
        ("elu", {"alpha": 0.7}),
        ("selu", {}),
        ("silu", {}),
        ("softplus", {"beta": 1.3, "threshold": 1}),
        ("hardtanh", {"min_val": -0.2, "max_val": 0.4}),
        ("hardsigmoid", {}),
        ("hardswish", {}),
    ):
        check(
            f"{name}/backward",
            getattr(functional, name),
            (torch.tensor([[-4.2, -2.5, -0.4], [0.3, 2.5, 4.2]], requires_grad=True),),
            device,
            kwargs=options,
            gradients=True,
        )
    for left, right in (
        (torch.float16, torch.float32),
        (torch.int16, torch.int64),
        (torch.int32, torch.float32),
    ):
        check(
            f"add/promotion={left}+{right}",
            torch.add,
            (sample((2, 3), left), sample((1, 3), right)),
            device,
        )

    x = sample((3, 4)).requires_grad_()
    for name, function in (
        ("transpose", lambda a: a.transpose(0, 1)),
        ("permute", lambda a: a.unsqueeze(0).permute(2, 0, 1)),
        ("squeeze", lambda a: a.unsqueeze(1).squeeze(1)),
        ("unbind", lambda a: a.unbind(0)),
        ("diagonal", lambda a: a.diagonal(offset=1)),
        ("flip", lambda a: a.flip((0, 1))),
        ("triu", lambda a: a.triu(diagonal=1)),
        ("tril", lambda a: a.tril(diagonal=-1)),
        ("diag/vector-to-matrix", lambda a: a[0].diag(diagonal=1)),
        ("diag/matrix-to-vector", lambda a: a.diag(diagonal=-1)),
        ("noncontiguous-pointwise", lambda a: a.T.sin()),
        ("slice", lambda a: a[1:, ::2]),
        ("narrow", lambda a: a.narrow(1, 1, 2)),
        ("as_strided/offset", lambda a: a.as_strided((2, 2), (4, 1), storage_offset=1)),
        ("select", lambda a: a.select(0, 1)),
        ("reshape", lambda a: a.T.reshape(2, 6)),
        ("flatten", lambda a: a.flatten()),
        ("expand", lambda a: a[:1].expand(3, -1)),
        ("repeat", lambda a: a.repeat(2, 1)),
        ("cat", lambda a: torch.cat((a, a + 2), dim=1)),
        ("stack", lambda a: torch.stack((a, a * 2), dim=0)),
        ("split", lambda a: a.split([1, 3], dim=1)),
        ("chunk", lambda a: a.chunk(2, dim=0)),
        ("unfold", lambda a: a.unfold(1, 2, 1)),
    ):
        check(f"{name}/values-and-backward", function, (x,), device, gradients=True)
    check(
        "transpose_/metadata",
        lambda a: a.transpose_(0, 1),
        (x.detach(),),
        device,
        inplace=True,
    )
    check(
        "resize_/same-numel",
        lambda a: a.resize_(2, 6),
        (x.detach(),),
        device,
        inplace=True,
    )
    check(
        "copy_/noncontiguous",
        lambda a, b: a.copy_(b.T),
        (torch.zeros(3, 4), sample((4, 3))),
        device,
        inplace=True,
    )
    for shape in ((), (0, 3), (2, 1, 3)):
        value = sample(shape)
        check(f"copy/shape={shape}", lambda a: a.clone(), (value,), device)
        check(f"add/shape={shape}", torch.add, (value, 1), device)

    indices = torch.tensor([[0, 2, 1], [1, 1, 0]], dtype=torch.int64)
    check(
        "gather/backward",
        torch.gather,
        (x[:2].detach().requires_grad_(), 1, indices),
        device,
        gradients=True,
    )
    check(
        "index_select/backward",
        torch.index_select,
        (x, 0, torch.tensor([2, 0, 2])),
        device,
        gradients=True,
    )
    check(
        "embedding/repeated-indices",
        functional.embedding,
        (torch.tensor([[1, 3, 1], [0, 4, 2]]), sample((5, 4)).requires_grad_()),
        device,
        kwargs={"padding_idx": 0},
        gradients=True,
    )
    check(
        "masked_select/backward",
        torch.masked_select,
        (x, x > 0),
        device,
        gradients=True,
    )
    check(
        "masked_fill",
        lambda a, m: a.masked_fill_(m, -2),
        (x.detach(), x > 0),
        device,
        inplace=True,
    )
    check(
        "masked_scatter",
        lambda a, m, v: a.masked_scatter_(m, v),
        (x.detach(), x > 0, sample((int((x > 0).sum()),))),
        device,
        inplace=True,
    )
    check("nonzero", torch.nonzero, (x > 0,), device)
    check(
        "unique/inverse-and-counts",
        torch.unique,
        (torch.tensor([[3, 1, 3], [-2, 1, 0]]),),
        device,
        kwargs={"sorted": True, "return_inverse": True, "return_counts": True},
    )
    check(
        "index_put/accumulate",
        lambda a, i, v: a.index_put_((i,), v, accumulate=True),
        (torch.zeros(4), torch.tensor([1, 1, 3]), torch.tensor([2.0, -0.5, 4.0])),
        device,
        inplace=True,
    )
    check(
        "scatter_add/repeated-indices/backward",
        torch.scatter_add,
        (sample((2, 3)).requires_grad_(), 1, indices, sample((2, 3)).requires_grad_()),
        device,
        gradients=True,
    )
    for invert in (False, True):
        check(
            f"isin/duplicates/invert={invert}",
            torch.isin,
            (torch.tensor([3, -1, 3, 7, 0]), torch.tensor([0, 3, 3, 5])),
            device,
            kwargs={"invert": invert},
        )
    for reduce in ("add", "multiply"):
        destination = torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
        source = torch.tensor([[0.5, 2.0, -1.0], [3.0, -2.0, 0.25]])
        check(
            f"scatter.reduce_out/{reduce}",
            torch.ops.aten.scatter.reduce_out,
            (destination, 1, indices, source),
            device,
            kwargs={"reduce": reduce, "out": output_like(destination)},
        )
        check(
            f"scatter.value_reduce_out/{reduce}",
            torch.ops.aten.scatter.value_reduce_out,
            (destination, 1, indices, 2.0),
            device,
            kwargs={"reduce": reduce, "out": output_like(destination)},
        )

    unique_indices = torch.tensor([[0, 2, 1], [1, 2, 0]])
    for source in (sample((2, 3)), -0.7):
        check(
            f"scatter/assignment/{type(source).__name__}",
            torch.scatter,
            (sample((2, 3)), 1, unique_indices, source),
            device,
        )
    for name in ("isfinite", "isinf", "isnan"):
        check(
            name,
            getattr(torch, name),
            (torch.tensor([-float("inf"), -1.0, 0.0, float("inf"), float("nan")]),),
            device,
            equal_nan=True,
        )

    values = torch.tensor(
        [[3.0, -2.0, 5.0, 0.5], [7.0, 1.0, -4.0, 2.0]], requires_grad=True
    )
    for name in ("min", "max"):
        check(
            f"{name}/values-indices-backward",
            getattr(torch, name),
            (values,),
            device,
            kwargs={"dim": 1, "keepdim": True},
            gradients=True,
        )
    check(
        "sort/unique-values",
        torch.sort,
        (values,),
        device,
        kwargs={"dim": 0},
        gradients=True,
    )
    check(
        "topk/unique-values",
        torch.topk,
        (values, 2),
        device,
        kwargs={"dim": 1},
        gradients=True,
    )
    check(
        "topk/smallest",
        torch.topk,
        (values, 2),
        device,
        kwargs={"dim": 1, "largest": False},
        gradients=True,
    )
    tied = torch.tensor([[4.0, 4.0, 1.0, 4.0], [-1.0, 2.0, 2.0, -3.0]])
    check("argmax/first-tie", torch.argmax, (tied, 1), device)
    check("sort/stable-ties", torch.sort, (tied,), device, kwargs={"stable": True})
    selected, positions = torch.topk(tied.to(device), 2, dim=1)
    if (
        positions.device != device
        or positions.dtype is not torch.int64
        or positions.shape != (2, 2)
    ):
        raise AssertionError("topk returned invalid index metadata")
    index_cpu = positions.cpu()
    if not ((index_cpu >= 0) & (index_cpu < tied.shape[1])).all():
        raise AssertionError("topk returned an out-of-range index")
    if (index_cpu[:, 0] == index_cpu[:, 1]).any():
        raise AssertionError("topk selected the same input position twice")
    expected_values = torch.topk(tied, 2, dim=1).values
    assert_result(selected, expected_values, device)
    torch.testing.assert_close(tied.gather(1, index_cpu), selected.cpu())
    print("PASS topk/unspecified-tie-order with valid distinct indices", flush=True)

    from torchvision.ops import nms

    boxes = torch.tensor(
        [[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0], [4.0, 4.0, 5.0, 5.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    for threshold in (0.2, 0.5):
        check(f"nms/threshold={threshold}", nms, (boxes, scores, threshold), device)


if __name__ == "__main__":
    run(check_ops)
