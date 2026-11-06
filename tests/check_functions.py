"""Check the six fused model functions against unfused CPU operations."""

import torch
from common import check, run, sample
from torch.nn import functional


def attention(query, key, value, embed_dim, num_heads, past_key, past_value, mask=None):
    batch, length, _ = query.shape
    query = query.reshape(batch, length, num_heads, -1).transpose(1, 2)
    key = key.reshape(batch, -1, num_heads, embed_dim // num_heads).transpose(1, 2)
    value = value.reshape(batch, -1, num_heads, embed_dim // num_heads).transpose(1, 2)
    if past_key is not None:
        key = torch.cat((past_key, key), dim=2)
        value = torch.cat((past_value, value), dim=2)
    output = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask.bool() if mask is not None else None
    )
    return output.transpose(1, 2).reshape(batch, length, embed_dim), key, value


def rotary(query, key, cos, sin, position, length, width):
    cos = cos[..., position : position + length, :width]
    sin = sin[..., position : position + length, :width]

    def rotate(value):
        head = value[..., :width]
        left, right = head.chunk(2, dim=-1)
        rotated = head * cos + torch.cat((-right, left), dim=-1) * sin
        return torch.cat((rotated, value[..., width:]), dim=-1)

    return rotate(query), rotate(key)


def phi2(value, weight1, weight2, bias1=None, bias2=None):
    # Phi-2 uses GELU-new; the weight order follows Microsoft's DirectML model sample.
    hidden = functional.gelu(
        functional.linear(value, weight1, bias1), approximate="tanh"
    )
    return functional.linear(hidden, weight2, bias2)


def llama(value, weight1, weight2, weight3, bias1=None, bias2=None, bias3=None):
    # The public function takes the gate, up and down projections in that order (w1, w3, w2).
    hidden = functional.silu(functional.linear(value, weight1, bias1))
    hidden = hidden * functional.linear(value, weight2, bias2)
    return functional.linear(hidden, weight3, bias3)


def phi3(value, weight1, weight2, bias1=None, bias2=None):
    gate, up = functional.linear(value, weight1, bias1).chunk(2, dim=-1)
    return functional.linear(functional.silu(gate) * up, weight2, bias2)


def check_functions(device):
    """Check numerical results, optional arguments, caches and rotary positions."""
    import torch_directml as dml

    for dtype in (torch.float32, torch.float16):
        query = sample((1, 3, 8), dtype)
        key = query.flip(-1) * 0.7 - 0.1
        value = query.flip(1) * -0.4 + 0.3
        for cached in (False, True):
            past_key = sample((1, 2, 2, 4), dtype) if cached else None
            past_value = past_key * -0.3 + 0.1 if cached else None
            for masked in (False, True):
                # The upstream Llama sample passes an int32 [batch, heads, query, key] keep mask.
                mask = torch.ones((1, 2, 3, 5 if cached else 3), dtype=torch.int32)
                mask = mask.tril(diagonal=2 if cached else 0) if masked else None
                check(
                    f"attention/{dtype}/cached={cached}/masked={masked}",
                    dml.multi_head_attention,
                    (query, key, value, 8, 2, past_key, past_value, mask),
                    device,
                    reference=attention,
                )

        for width in (4, 8):
            query = sample((1, 2, 3, 8), dtype)
            key = sample((1, 1, 3, 8), dtype).flip(-1)
            angles = torch.arange(8).outer(torch.linspace(0.2, 0.8, width // 2))
            angles = torch.cat((angles, angles), dim=-1)[None, None]
            # Revisit a cached configuration with new Tensor data after using another configuration.
            for trial, position in enumerate((0, 2, 0)):
                check(
                    f"rotary/{dtype}/width={width}/position={position}",
                    dml.apply_rotary_position_emb,
                    (
                        query if trial < 2 else query.flip(-1) * 0.8,
                        key if trial < 2 else key.flip(-1) + 0.1,
                        angles.cos().to(dtype),
                        angles.sin().to(dtype),
                        position,
                        3,
                        width,
                    ),
                    device,
                    reference=rotary,
                )

        inputs = sample((1, 2, 8), dtype)
        gate, up = sample((16, 8), dtype), sample((16, 8), dtype).flip(-1) * -0.6
        down = sample((8, 16), dtype) * 0.4
        for biases in (False, True):
            bias1 = sample((16,), dtype) if biases else None
            bias2 = sample((8,), dtype) * -0.4 if biases else None
            bias_up = sample((16,), dtype) * 0.6 if biases else None
            check(
                f"mlp_phi2/{dtype}/bias={biases}",
                dml.mlp_phi2,
                (inputs, gate, down, bias1, bias2),
                device,
                reference=phi2,
            )
            check(
                f"mlp_llama/{dtype}/bias={biases}",
                dml.mlp_llama,
                (inputs, gate, up, down, bias1, bias_up, bias2),
                device,
                reference=llama,
            )
            combined_bias = torch.cat((bias1, bias_up)) if biases else None
            check(
                f"mlp_phi3/{dtype}/bias={biases}",
                dml.mlp_phi3,
                (inputs, torch.cat((gate, up)), down, combined_bias, bias2),
                device,
                reference=phi3,
            )

    # Upstream RMSNorm explicitly uses FP32. Small inputs make epsilon observable.
    inputs = sample((2, 3, 8)) * 0.01
    weight = torch.linspace(-0.7, 1.4, 8)
    for trial, epsilon in enumerate((1e-5, 1e-3, 1e-5)):
        check(
            f"rmsnorm/epsilon={epsilon}/data={trial}",
            dml.rmsnorm,
            (
                inputs if trial < 2 else inputs.flip(-1) * 0.8,
                weight if trial < 2 else weight.flip(0) * 0.6,
                epsilon,
            ),
            device,
            reference=lambda x, w, eps: (
                x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w
            ),
        )


if __name__ == "__main__":
    run(check_functions)
