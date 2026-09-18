"""HF safetensors → the checkpoint layout ExecuTorch's llama_transformer loads.

Key names match ExecuTorch 1.4.0's own converters (examples/models/{qwen3,qwen2_5,
smollm2}/convert_weights.py). Those load everything through torchtune or into one dict
and only handle single-file or specific checkpoints; this one streams shards, handles
tied embeddings from the config, and fails on any tensor it does not know rather than
silently dropping it.

The checkpoint's file name must not contain "int8" or "8da4w": examples/models/llama/
model.py switches to pre-quantized loading when the path contains either string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# Per-layer HF suffix → ExecuTorch suffix, shared by every supported family.
_LAYER_KEYS = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    # ExecuTorch applies the activation to w1, as HF does to gate_proj.
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}
_QKV_BIAS_KEYS = {
    "self_attn.q_proj.bias": "attention.wq.bias",
    "self_attn.k_proj.bias": "attention.wk.bias",
    "self_attn.v_proj.bias": "attention.wv.bias",
}
_QK_NORM_KEYS = {
    "self_attn.q_norm.weight": "attention.q_norm_fn.weight",
    "self_attn.k_norm.weight": "attention.k_norm_fn.weight",
}
_TOP_KEYS = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}
# LFM2 keeps the final norm under another name and has no lm_head: its embeddings are tied.
_LFM2_TOP_KEYS = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.embedding_norm.weight": "norm.weight",
}
# LFM2 names the attention output and the QK norms differently, carries the block norm as
# operator_norm, and writes the feed-forward and the short convolution in ExecuTorch's own
# names already (examples/models/lfm2/convert_weights.py).
_LFM2_LAYER_KEYS = {
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.out_proj.weight": "attention.wo.weight",
    "self_attn.q_layernorm.weight": "attention.q_norm_fn.weight",
    "self_attn.k_layernorm.weight": "attention.k_norm_fn.weight",
    "operator_norm.weight": "attention_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "feed_forward.w1.weight": "feed_forward.w1.weight",
    "feed_forward.w2.weight": "feed_forward.w2.weight",
    "feed_forward.w3.weight": "feed_forward.w3.weight",
    "conv.conv.weight": "conv.conv.weight",
    "conv.out_proj.weight": "conv.out_proj.weight",
    "conv.in_proj.weight": "conv.in_proj.weight",
}
# One HF tensor that becomes three: the short convolution's fused input projection is stored
# as B, C and x stacked on dim 0, and ExecuTorch's ShortConv reads them apart.
SPLIT_THREE = {"lfm2": ("conv.in_proj.weight", ("conv.B_proj.weight", "conv.C_proj.weight", "conv.x_proj.weight"))}
# Buffers some checkpoints carry that the model recomputes.
_IGNORED = re.compile(r"(rotary_emb\.inv_freq|\.masked_bias|\.attn\.bias)$")

CONVERTERS = {
    # family converter key → (extra per-layer keys, un-permute q/k for Meta RoPE)
    "qwen3": ({**_QK_NORM_KEYS}, False),
    "qwen2": ({**_QKV_BIAS_KEYS}, False),
    "llama": ({}, True),
    # LFM2 shares none of the per-layer names, so its map replaces rather than extends the
    # common one; every layer is a short convolution or an attention block, never both.
    "lfm2": (_LFM2_LAYER_KEYS, False),
}
# Converters whose per-layer map stands alone instead of extending _LAYER_KEYS.
_STANDALONE = {"lfm2"}
# A conv layer and an attention layer carry different tensors, so the per-layer count that
# the other families check cannot apply; LFM2 is checked by total instead.
_MIXED_LAYERS = {"lfm2"}


def unpermute(weight, n_heads: int):
    """Inverse of the q/k permutation HF's Llama conversion applies for rotate_half RoPE.

    HF: w.view(n_heads, head_dim // 2, 2, dim).transpose(1, 2).reshape(out, dim)
    """
    out_features, in_features = weight.shape
    return (
        weight.view(n_heads, 2, out_features // n_heads // 2, in_features)
        .transpose(1, 2)
        .reshape(out_features, in_features)
    )


def _shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [model_dir / name for name in sorted(set(weight_map.values()))]
    single = model_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors checkpoint in {model_dir}")


def map_key(hf_key: str, converter: str) -> str | None:
    """ExecuTorch name for an HF tensor, None for ignorable buffers, KeyError if unknown."""
    extra, _ = CONVERTERS[converter]
    top = _LFM2_TOP_KEYS if converter == "lfm2" else _TOP_KEYS
    if hf_key in top:
        return top[hf_key]
    if _IGNORED.search(hf_key):
        return None
    match = _LAYER.match(hf_key)
    if match:
        index, suffix = match.groups()
        common = {} if converter in _STANDALONE else _LAYER_KEYS
        target = common.get(suffix) or extra.get(suffix)
        if target:
            return f"layers.{index}.{target}"
    raise KeyError(f"tensor {hf_key!r} has no mapping for converter {converter!r}")


def convert(model_dir: Path, output: Path, converter: str, config: dict) -> dict:
    """Write the converted checkpoint to ``output``; returns a summary for the report."""
    import torch
    from safetensors import safe_open

    if "int8" in output.name or "8da4w" in output.name:
        raise ValueError(f"checkpoint name {output.name!r} would trigger pre-quantized loading")
    _, permute = CONVERTERS[converter]
    n_heads = int(config["num_attention_heads"])
    n_kv_heads = int(config.get("num_key_value_heads") or n_heads)

    state: dict[str, torch.Tensor] = {}
    for shard in _shards(model_dir):
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                target = map_key(key, converter)
                if target is None:
                    continue
                tensor = f.get_tensor(key)
                split = SPLIT_THREE.get(converter)
                if split and target.endswith(split[0]):
                    # The fused projection holds B, C and x stacked on dim 0, in that order.
                    head = target[: -len(split[0])]
                    for name, part in zip(split[1], torch.chunk(tensor, 3, dim=0), strict=True):
                        state[head + name] = part
                    continue
                if permute and target.endswith("attention.wq.weight"):
                    tensor = unpermute(tensor, n_heads)
                elif permute and target.endswith("attention.wk.weight"):
                    tensor = unpermute(tensor, n_kv_heads)
                state[target] = tensor

    tied = "output.weight" not in state
    if tied:
        if not config.get("tie_word_embeddings", True):
            raise KeyError("checkpoint has no lm_head.weight but config says embeddings are untied")
        state["output.weight"] = state["tok_embeddings.weight"]

    n_layers = int(config["num_hidden_layers"])
    per_layer = sum(1 for k in state if k.startswith("layers."))
    if converter in _MIXED_LAYERS:
        # Every layer is one kind or the other, so the count is checked per index rather
        # than against a single width: a missing tensor still shows up as a short layer.
        seen: dict[str, int] = {}
        for key in state:
            if key.startswith("layers."):
                seen[key.split(".")[1]] = seen.get(key.split(".")[1], 0) + 1
        if len(seen) != n_layers:
            raise ValueError(f"expected {n_layers} layers, found {len(seen)}")
        widths = sorted(set(seen.values()))
        if not widths or min(widths) < 6:
            raise ValueError(f"a layer is missing tensors: widths {widths}")
    else:
        expected_per_layer = len(_LAYER_KEYS) + len(CONVERTERS[converter][0])
        if per_layer != n_layers * expected_per_layer:
            raise ValueError(f"expected {n_layers} layers x {expected_per_layer} tensors, found {per_layer}")

    dtypes = sorted({str(t.dtype) for t in state.values()})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    return {"tensors": len(state), "tied_embeddings": tied, "dtypes": dtypes}
