"""Model families: which HF architectures each backend can export, and how.

For XNNPACK, ExecuTorch 1.4.0's ``export_llm`` builds every model from one transformer
definition (examples/models/llama/llama_transformer.py). ``base.model_class`` only picks
the example directory and a few class-specific behaviours; the params file defines the
architecture. So a new size or finetune of a known architecture is exported by
generating the params from its HF config.json, not by waiting for ExecuTorch to list it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pipeline.sizing import Architecture

# HF config keys that mean the checkpoint is a mixture of experts.
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_intermediate_size")

# Llama 3.1+ RoPE scaling that ExecuTorch's non-HF RoPE implements: rope.apply_scaling
# hard-codes low_freq_factor 1 and an 8192-token original context, and model.py forces
# rope_scale_factor 32 for every scaled-RoPE model class except llama3/llama3_1.
_LLAMA32_ROPE = {
    "rope_type": "llama3",
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


class UnsupportedModel(Exception):
    """The checkpoint's architecture is outside what the recipe can export correctly."""


@dataclass(frozen=True)
class Family:
    key: str
    architectures: tuple[str, ...]
    # Backend → None when supported, or the reason it is not (yet).
    unsupported: dict[str, str] = field(default_factory=dict)

    def supports(self, backend: str) -> bool:
        return backend not in self.unsupported


_NOT_IN_EXPORT_LLM = (
    "not in ExecuTorch 1.4.0 export_llm's model list; needs a validated export path "
    "(optimum-executorch or params support) before it is published"
)
_MTK_LLAMA = (
    "not validated on ExecuTorch 1.4.0's MediaTek scripts: they read rope_scaling['type'], "
    "Llama 3.2 configs carry rope_type llama3, and SmolLM2 needs its tokenizer class chosen"
)
_MTK_GEMMA3 = "not validated on the MediaTek scripts (they expect model_type gemma3, HF has gemma3_text)"
_MTK_NO_MODEL = "no model definition for this architecture in ExecuTorch 1.4.0's examples/mediatek"
_LFM2_NO_VULKAN = (
    "the Vulkan delegate has no kernel for the short convolution, and an LFM2.5 file that "
    "lowers to it segfaults in the 1.4.0 runtime at the first prefill"
)
_LFM2_NO_QNN = "not in ExecuTorch 1.4.0's Qualcomm SUPPORTED_LLM_MODELS registry"

FAMILIES: tuple[Family, ...] = (
    Family("qwen3", ("Qwen3ForCausalLM",)),
    # LFM2 is a short-convolution hybrid: only some layers attend. ExecuTorch 1.4.0 has it
    # under examples/models/lfm2 for XNNPACK, and neither vendor NPU path has a definition
    # for it, nor does the Vulkan recipe cover the convolution.
    Family(
        "lfm2",
        ("Lfm2ForCausalLM",),
        {"vulkan": _LFM2_NO_VULKAN, "qnn": _LFM2_NO_QNN, "mtk": _MTK_NO_MODEL},
    ),
    Family("qwen2_5", ("Qwen2ForCausalLM",)),
    Family("llama", ("LlamaForCausalLM",), {"mtk": _MTK_LLAMA}),
    Family(
        "gemma3",
        ("Gemma3ForCausalLM",),
        {"xnnpack": _NOT_IN_EXPORT_LLM, "vulkan": _NOT_IN_EXPORT_LLM, "mtk": _MTK_GEMMA3},
    ),
    Family(
        "smollm3",
        ("SmolLM3ForCausalLM",),
        {"xnnpack": _NOT_IN_EXPORT_LLM, "vulkan": _NOT_IN_EXPORT_LLM, "mtk": _MTK_NO_MODEL},
    ),
)

# ExecuTorch 1.4.0's Qualcomm LLM scripts (examples/qualcomm/oss_scripts/llama, the
# SUPPORTED_LLM_MODELS registry) export a fixed list of checkpoints, each with its own
# quantization recipe. These are the ones in the watched orgs: source repo -> --decoder_model.
QNN_DECODERS = {
    "Qwen/Qwen3-0.6B": "qwen3-0_6b",
    "Qwen/Qwen3-1.7B": "qwen3-1_7b",
    "Qwen/Qwen2.5-0.5B": "qwen2_5-0_5b",
    "Qwen/Qwen2.5-1.5B": "qwen2_5-1_5b",
    "google/gemma-3-1b-it": "gemma3-1b",
    "HuggingFaceTB/SmolLM2-135M-Instruct": "smollm2_135m",
    "HuggingFaceTB/SmolLM3-3B": "smollm3-3b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama3_2-1b_instruct",
    "meta-llama/Llama-3.2-3B-Instruct": "llama3_2-3b_instruct",
}
# Registry entries with no repo_id: the script needs Meta's original checkpoint, params and
# tokenizer passed in, which meta-llama's HF repos carry under original/.
QNN_META_CHECKPOINT = {"llama3_2-1b_instruct", "llama3_2-3b_instruct"}
# The registry's params_path for the others, relative to the ExecuTorch source tree. The
# wheel does not ship these .json files; copies live in third_party/executorch.
QNN_PARAMS = {
    "qwen3-0_6b": "examples/models/qwen3/config/0_6b_config.json",
    "qwen3-1_7b": "examples/models/qwen3/config/1_7b_config.json",
    "qwen2_5-0_5b": "examples/models/qwen2_5/config/0_5b_config.json",
    "qwen2_5-1_5b": "examples/models/qwen2_5/config/1_5b_config.json",
    "gemma3-1b": "examples/models/gemma3/config/1b_config.json",
    "smollm2_135m": "examples/models/smollm2/135M_config.json",
    "smollm3-3b": "examples/models/smollm3/3b_config.json",
}
QNN_UNLISTED = "no entry for this checkpoint in ExecuTorch 1.4.0's Qualcomm LLM scripts"


def qnn_decoder(model_id: str) -> str | None:
    return QNN_DECODERS.get(model_id)


@dataclass(frozen=True)
class MtkPlan:
    script: str  # examples/mediatek/model_export_scripts/<script>
    preformatter: str  # examples/mediatek/aot_utils/llm_utils/preformatter_templates/<name>
    num_chunks: int


# ExecuTorch 1.4.0 examples/mediatek: export script and chat template per family, the pair
# its shell_scripts/export_*.sh use. The model definition comes from config.json's
# model_type (aot_utils/llm_utils/utils.py resolve_model_classes).
_MTK_SCRIPTS = {
    "qwen3": ("qwen.py", "qwen3.json", "qwen3"),
    "qwen2_5": ("qwen.py", "qwen.json", "qwen2"),
}


def mtk_chunks(n_layers: int, max_chunks: int) -> int:
    """Most chunks up to ``max_chunks`` that split the layers evenly (the scripts require it)."""
    return next(n for n in range(min(max_chunks, n_layers), 0, -1) if n_layers % n == 0)


def mtk_plan(family: Family, config: dict, max_chunks: int) -> MtkPlan:
    if family.key not in _MTK_SCRIPTS:
        raise UnsupportedModel(f"no MediaTek export script mapped for {family.key}")
    script, preformatter, model_type = _MTK_SCRIPTS[family.key]
    c = text_config(config)
    if c.get("model_type") != model_type:
        raise UnsupportedModel(f"model_type {c.get('model_type')!r}, the MediaTek {script} path expects {model_type!r}")
    if rope_scaling(c) is not None:
        raise UnsupportedModel("RoPE scaling is not validated on the MediaTek scripts")
    return MtkPlan(script, preformatter, mtk_chunks(int(c["num_hidden_layers"]), max_chunks))


BACKENDS = ("xnnpack", "vulkan", "qnn", "mtk")


def family_for(config: dict) -> Family | None:
    architectures = config.get("architectures") or []
    for family in FAMILIES:
        if any(a in family.architectures for a in architectures):
            return family
    return None


def is_moe(config: dict) -> bool:
    if any("moe" in a.lower() for a in config.get("architectures") or []):
        return True
    for key in _MOE_KEYS:
        value = config.get(key)
        if value not in (None, 0, False):
            return True
    return False


def text_config(config: dict) -> dict:
    return config.get("text_config") or config


def rope_theta(config: dict) -> float:
    if "rope_theta" in config:
        return float(config["rope_theta"])
    params = config.get("rope_parameters") or {}
    if "rope_theta" in params:
        return float(params["rope_theta"])
    raise UnsupportedModel("config has no rope_theta")


def rope_scaling(config: dict) -> dict | None:
    """The rope scaling block, or None for plain RoPE (transformers v4 and v5 spellings)."""
    scaling = config.get("rope_scaling") or config.get("rope_parameters")
    if not scaling:
        return None
    kind = scaling.get("rope_type") or scaling.get("type")
    if kind in (None, "default"):
        return None
    return scaling


def architecture(config: dict, total_params: int) -> Architecture:
    c = text_config(config)
    n_heads = int(c["num_attention_heads"])
    dim = int(c["hidden_size"])
    return Architecture(
        n_layers=int(c["num_hidden_layers"]),
        n_heads=n_heads,
        n_kv_heads=int(c.get("num_key_value_heads") or n_heads),
        head_dim=int(c.get("head_dim") or dim // n_heads),
        vocab_size=int(c["vocab_size"]),
        dim=dim,
        intermediate=int(c["intermediate_size"]),
        total_params=int(total_params),
        # transformers' PretrainedConfig default, the same one convert.py assumes.
        tied_embeddings=bool(c.get("tie_word_embeddings", True)),
    )


def lfm2_hidden_dim(c: dict) -> int:
    """LFM2's feed-forward width, which is not the config's ``intermediate_size``.

    Liquid's block auto-adjusts it: two thirds of ``block_ff_dim``, times
    ``block_ffn_dim_multiplier``, rounded up to ``block_multiple_of``. For LFM2.5-1.2B that
    turns 12,288 into 8,192, and the published weights are 8,192 wide, so reading
    ``intermediate_size`` instead would build a model that cannot load its own checkpoint.
    When ``block_auto_adjust_ff_dim`` is false, as in LFM2.5-2.6B, the config's value stands
    (10,752 there, matching its weights).
    """
    if not c.get("block_auto_adjust_ff_dim"):
        return int(c["intermediate_size"])
    ff = int(2 * int(c["block_ff_dim"]) / 3)
    ff = int(float(c.get("block_ffn_dim_multiplier") or 1) * ff)
    multiple = int(c.get("block_multiple_of") or 256)
    return multiple * ((ff + multiple - 1) // multiple)


def _common_params(c: dict) -> dict:
    n_heads = int(c["num_attention_heads"])
    dim = int(c["hidden_size"])
    return {
        "dim": dim,
        "ffn_dim_multiplier": 1,
        "hidden_dim": int(c["intermediate_size"]),
        "n_heads": n_heads,
        "head_dim": int(c.get("head_dim") or dim // n_heads),
        "n_kv_heads": int(c.get("num_key_value_heads") or n_heads),
        "n_layers": int(c["num_hidden_layers"]),
        "norm_eps": float(c.get("rms_norm_eps") or c["norm_eps"]),
        "rope_theta": rope_theta(c),
        "vocab_size": int(c["vocab_size"]),
    }


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise UnsupportedModel(reason)


def _check_common(c: dict) -> None:
    _require(c.get("hidden_act", "silu") == "silu", f"activation {c.get('hidden_act')!r} is not silu")
    _require(not c.get("use_sliding_window"), "sliding-window attention is enabled")
    _require(not c.get("mlp_bias"), "MLP biases are not supported by the recipe")


@dataclass(frozen=True)
class XnnpackPlan:
    """What export_llm needs besides the shared recipe."""

    model_class: str
    params: dict
    converter: str  # key into pipeline.convert.CONVERTERS


def xnnpack_plan(family: Family, config: dict) -> XnnpackPlan:
    """ExecuTorch params and model class for this checkpoint, or UnsupportedModel."""
    if not family.supports("xnnpack"):
        raise UnsupportedModel(family.unsupported["xnnpack"])
    c = text_config(config)
    _check_common(c)
    params = _common_params(c)

    if family.key == "qwen3":
        _require(rope_scaling(c) is None, "RoPE scaling (e.g. YaRN) is not supported")
        _require(not c.get("attention_bias"), "Qwen3 with attention biases is not supported")
        params.update(
            use_scaled_rope=False,
            use_hf_rope=True,
            attention_qkv_bias=False,
            use_qk_norm=True,
            qk_norm_before_rope=True,
        )
        return XnnpackPlan("qwen3_1_7b", params, "qwen3")

    if family.key == "lfm2":
        _require(rope_scaling(c) is None, "RoPE scaling (e.g. YaRN) is not supported")
        layer_types = c.get("layer_types")
        _require(bool(layer_types), "LFM2 config has no layer_types")
        _require(
            len(layer_types) == int(c["num_hidden_layers"]),
            "layer_types does not match num_hidden_layers",
        )
        _require(
            set(layer_types) <= {"conv", "full_attention"},
            f"unknown LFM2 layer types: {sorted(set(layer_types))}",
        )
        _require(not c.get("conv_bias"), "LFM2 with a convolution bias is not supported")
        params.update(
            hidden_dim=lfm2_hidden_dim(c),
            use_scaled_rope=False,
            use_hf_rope=True,
            use_qk_norm=True,
            qk_norm_before_rope=True,
            layer_types=list(layer_types),
        )
        params.pop("head_dim", None)
        return XnnpackPlan("lfm2_5_1_2b", params, "lfm2")

    if family.key == "qwen2_5":
        _require(rope_scaling(c) is None, "RoPE scaling (e.g. YaRN) is not supported")
        params.update(use_scaled_rope=False, use_hf_rope=True, attention_qkv_bias=True)
        return XnnpackPlan("qwen2_5_1_5b", params, "qwen2")

    if family.key == "llama":
        _require(not c.get("attention_bias"), "Llama with attention biases is not supported")
        scaling = rope_scaling(c)
        if scaling is None:
            # Plain RoPE (SmolLM2). Weights are un-permuted to Meta's interleaved layout
            # and run through ExecuTorch's Meta RoPE, as torchtune's converter does.
            params.update(use_scaled_rope=False, use_hf_rope=False, attention_qkv_bias=False)
            return XnnpackPlan("smollm2", params, "llama")
        for key, expected in _LLAMA32_ROPE.items():
            actual = scaling.get(key, scaling.get("type") if key == "rope_type" else None)
            if isinstance(expected, float):
                ok = actual is not None and float(actual) == expected
            else:
                ok = actual == expected
            _require(ok, f"rope_scaling {key}={actual!r}, recipe only implements {expected!r}")
        params.update(use_scaled_rope=True, use_hf_rope=False, attention_qkv_bias=False)
        return XnnpackPlan("llama3_2", params, "llama")

    raise UnsupportedModel(f"no XNNPACK recipe for family {family.key}")
