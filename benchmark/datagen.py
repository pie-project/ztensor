import numpy as np
from typing import Dict, Union

import config


def generate_tensor_dict(
    total_size_mb: int,
    distribution: str = "mixed",
    data_style: str = "random",
) -> Dict[str, Union[np.ndarray, "torch.Tensor"]]:
    """
    Generates tensors summing to total_size_mb.
    distribution:
      - 'mixed': Realistic mix (some large weights, some small biases/metadata)
      - 'small': Many small tensors (1KB - 100KB). Stresses metadata/parsing.
      - 'large': Few large tensors (10MB - 100MB). Stresses raw BW.
    data_style:
      - 'random': Uniform random float32 (incompressible).
      - 'structured': Normal(0, 0.02) with block-sparsity (compressible, mimics real weights).
      - 'mixed_dtype': Realistic dtype mix (70% fp32, 20% fp16, 10% int8).

    Uses config.BACKEND setting to determine tensor type ('numpy' or 'torch').
    """
    style_tag = f", {data_style}" if data_style != "random" else ""
    print(
        f"  [{distribution.upper()}] Generating {total_size_mb}MB of synthetic data "
        f"(backend={config.BACKEND}{style_tag})..."
    )
    tensors = {}
    remaining_bytes = total_size_mb * 1024 * 1024
    i = 0

    while remaining_bytes > 0:
        # Determine dtype and element size for this tensor
        if data_style == "mixed_dtype":
            if i % 10 < 7:
                dtype = np.float32
                elem_size = 4
            elif i % 10 < 9:
                dtype = np.float16
                elem_size = 2
            else:
                dtype = np.int8
                elem_size = 1
        else:
            dtype = np.float32
            elem_size = 4

        # Determine shape based on distribution
        if distribution == "mixed":
            if remaining_bytes > 100 * 1024 * 1024:
                shape = (5000, 5000)
            elif remaining_bytes > 10 * 1024 * 1024:
                shape = (1000, 2500)
            else:
                elems = remaining_bytes // elem_size
                shape = (elems,)

        elif distribution == "large":
            target_bytes = 50 * 1024 * 1024
            if remaining_bytes < target_bytes:
                target_bytes = remaining_bytes
            elems = target_bytes // elem_size
            shape = (elems,)

        elif distribution == "small":
            target_bytes = 10 * 1024
            if remaining_bytes < target_bytes:
                target_bytes = remaining_bytes
            elems = target_bytes // elem_size
            shape = (elems,)

        elif distribution == "llama-1b":
            # Llama 3.2 1B architecture: hidden=2048, layers=16, kv_heads=8,
            # intermediate=8192, vocab=128256.
            # Total size: ~2.5 GB in fp16. Ignores total_size_mb.
            llama_tensors = _generate_llama_1b(data_style)
            return llama_tensors

        if np.prod(shape) == 0:
            break

        # Generate data based on data_style
        if data_style == "random":
            t_np = np.random.randn(*shape).astype(np.float32)
        elif data_style == "structured":
            t_np = np.random.normal(0.0, 0.02, size=shape).astype(np.float32)
            # Add block-sparsity: zero out ~20% of rows for realistic structure
            if len(shape) == 2 and shape[0] > 10:
                zero_rows = np.random.choice(
                    shape[0], size=shape[0] // 5, replace=False
                )
                t_np[zero_rows] = 0.0
        elif data_style == "mixed_dtype":
            if dtype == np.int8:
                t_np = np.random.randint(-128, 127, size=shape, dtype=np.int8)
            else:
                t_np = np.random.normal(0.0, 0.02, size=shape).astype(dtype)

        if config.BACKEND == "torch":
            import torch

            if dtype == np.int8:
                t = torch.from_numpy(t_np.copy())
            else:
                t = torch.from_numpy(t_np)
            remaining_bytes -= t.numel() * t.element_size()
        else:
            t = t_np
            remaining_bytes -= t.nbytes

        tensors[f"layer_{i}.weight"] = t
        i += 1
    return tensors


def _generate_llama_1b(data_style: str = "random") -> dict:
    """
    Generate tensors matching Llama 3.2 1B architecture shapes.
    hidden=2048, layers=16, heads=32, kv_heads=8, intermediate=8192, vocab=128256.
    All float16 (matching real HF checkpoints). Random data.
    ~1.24B params, ~2.5 GB in fp16.
    """
    H = 2048  # hidden_size
    KV = 512  # num_kv_heads * head_dim = 8 * 64
    I = 8192  # intermediate_size
    V = 128256  # vocab_size
    N = 16  # num_hidden_layers

    dtype = np.float16
    tensors = {}

    def _make(shape):
        if data_style == "structured":
            t = np.random.normal(0.0, 0.02, size=shape).astype(dtype)
            if len(shape) == 2 and shape[0] > 10:
                zero_rows = np.random.choice(
                    shape[0], size=shape[0] // 5, replace=False
                )
                t[zero_rows] = 0.0
            return t
        return np.random.randn(*shape).astype(dtype)

    # Embedding (tied with lm_head in Llama 3.2 1B, but we include both)
    tensors["model.embed_tokens.weight"] = _make((V, H))
    tensors["model.norm.weight"] = _make((H,))
    tensors["lm_head.weight"] = _make((V, H))

    # Per-layer weights (16 layers)
    for layer in range(N):
        p = f"model.layers.{layer}"
        # Self-attention (GQA: q_proj full size, k/v_proj reduced for 8 KV heads)
        tensors[f"{p}.self_attn.q_proj.weight"] = _make((H, H))
        tensors[f"{p}.self_attn.k_proj.weight"] = _make((KV, H))
        tensors[f"{p}.self_attn.v_proj.weight"] = _make((KV, H))
        tensors[f"{p}.self_attn.o_proj.weight"] = _make((H, H))
        # MLP
        tensors[f"{p}.mlp.gate_proj.weight"] = _make((I, H))
        tensors[f"{p}.mlp.up_proj.weight"] = _make((I, H))
        tensors[f"{p}.mlp.down_proj.weight"] = _make((H, I))
        # Layer norms
        tensors[f"{p}.input_layernorm.weight"] = _make((H,))
        tensors[f"{p}.post_attention_layernorm.weight"] = _make((H,))

    total_bytes = sum(t.nbytes for t in tensors.values())
    total_params = sum(t.size for t in tensors.values())
    print(
        f"  [LLAMA-1B] Generated {len(tensors)} tensors, "
        f"{total_params/1e9:.2f}B params, {total_bytes/1e9:.2f} GB ({data_style})"
    )

    return tensors
