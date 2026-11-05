import torch_directml_native

def multi_head_attention(query, key, value, embed_dim, num_heads, past_key, past_value, mask_opt=None):
    return torch_directml_native.multi_head_attention(query, key, value, embed_dim, num_heads, past_key, past_value, mask_opt)

def mlp_phi2(input, weight1, weight2, bias1=None, bias2=None):
    return torch_directml_native.mlp_phi2(input, weight1, weight2, bias1, bias2)

def mlp_llama(input, weight1, weight2, weight3, bias1=None, bias2=None, bias3=None):
    return torch_directml_native.mlp_llama(input, weight1, weight2, weight3, bias1, bias2, bias3)

def mlp_phi3(input, weight1, weight2, bias1=None, bias2=None):
    return torch_directml_native.mlp_phi3(input, weight1, weight2, bias1, bias2)

def rmsnorm(input, weight, eps):
    return torch_directml_native.rmsnorm(input, weight, eps)

def apply_rotary_position_emb(query, key, cos, sin, min_input_pos, sequence_length, embed_dim):
    return torch_directml_native.apply_rotary_position_emb(query, key, cos, sin, min_input_pos, sequence_length, embed_dim)
