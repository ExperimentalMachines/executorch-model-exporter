import copy
import dataclasses

import pytest
from conftest import hf_config, load_json

from pipeline import families


def plan_for(config):
    return families.xnnpack_plan(families.family_for(config), config)


def expected_params(name):
    """ExecuTorch's shipped params, with head_dim made explicit where it is implied."""
    params = load_json(name)
    params.setdefault("head_dim", params["dim"] // params["n_heads"])
    return params


@pytest.mark.parametrize(
    "config, reference",
    [
        ("qwen3-0.6b.config.json", "et-qwen3-0_6b.params.json"),
        ("qwen3-1.7b.config.json", "et-qwen3-1_7b.params.json"),
        ("qwen3-4b.config.json", "et-qwen3-4b.params.json"),
        ("qwen2.5-1.5b.config.json", "et-qwen2_5-1_5b.params.json"),
        ("smollm2-135m.config.json", "et-smollm2-135m.params.json"),
    ],
)
def test_generated_params_equal_executorchs_own(config, reference):
    assert plan_for(load_json(config)).params == expected_params(reference)


def test_model_classes_and_converters():
    assert (plan_for(hf_config("Qwen/Qwen3-1.7B")).model_class, plan_for(hf_config("Qwen/Qwen3-1.7B")).converter) == (
        "qwen3_1_7b",
        "qwen3",
    )
    qwen25 = plan_for(hf_config("Qwen/Qwen2.5-1.5B-Instruct"))
    assert (qwen25.model_class, qwen25.converter) == ("qwen2_5_1_5b", "qwen2")
    smol = plan_for(hf_config("HuggingFaceTB/SmolLM2-360M-Instruct"))
    assert (smol.model_class, smol.converter, smol.params["use_hf_rope"]) == ("smollm2", "llama", False)
    assert smol.params["head_dim"] == 64  # 960 / 15


def test_llama32_uses_scaled_meta_rope():
    plan = plan_for(hf_config("meta-llama/Llama-3.2-1B-Instruct"))
    assert plan.model_class == "llama3_2"  # model.py then sets rope_scale_factor = 32
    assert plan.params["use_scaled_rope"] is True
    assert plan.params["use_hf_rope"] is False
    assert plan.params["head_dim"] == 64
    assert plan.params["rope_theta"] == 500000.0


def test_other_llama_rope_scaling_is_refused():
    config = copy.deepcopy(hf_config("meta-llama/Llama-3.2-1B-Instruct"))
    config["rope_scaling"]["factor"] = 8.0  # Llama 3.1's factor
    with pytest.raises(families.UnsupportedModel, match="factor"):
        plan_for(config)


def test_yarn_is_refused():
    config = copy.deepcopy(hf_config("Qwen/Qwen3-1.7B"))
    config["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    with pytest.raises(families.UnsupportedModel, match="RoPE scaling"):
        plan_for(config)


def test_sliding_window_is_refused():
    config = copy.deepcopy(hf_config("Qwen/Qwen2.5-1.5B-Instruct"))
    config["use_sliding_window"] = True
    with pytest.raises(families.UnsupportedModel, match="sliding"):
        plan_for(config)


def test_transformers_v5_rope_parameters():
    config = copy.deepcopy(hf_config("Qwen/Qwen3-1.7B"))
    del config["rope_theta"]
    config.pop("rope_scaling", None)
    config["rope_parameters"] = {"rope_type": "default", "rope_theta": 1000000.0}
    assert plan_for(config).params["rope_theta"] == 1000000.0


def test_moe_detection():
    assert families.is_moe(hf_config("Qwen/Qwen3-30B-A3B"))
    for model_id in ("Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-1B-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct"):
        assert not families.is_moe(hf_config(model_id)), model_id


def test_gemma3_has_a_family_but_no_xnnpack_recipe_yet():
    family = families.family_for(hf_config("google/gemma-3-1b-it"))
    assert family.key == "gemma3"
    assert not family.supports("xnnpack")
    with pytest.raises(families.UnsupportedModel):
        families.xnnpack_plan(family, hf_config("google/gemma-3-1b-it"))


def _lfm2_config(**overrides):
    """LFM2.5-1.2B-Instruct's config, trimmed to what the plan reads."""
    config = {
        "architectures": ["Lfm2ForCausalLM"],
        "hidden_size": 2048,
        "intermediate_size": 12288,
        "block_ff_dim": 12288,
        "block_auto_adjust_ff_dim": True,
        "block_ffn_dim_multiplier": 1.0,
        "block_multiple_of": 256,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 4,
        "norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "vocab_size": 65536,
        "conv_bias": False,
        "layer_types": ["conv", "conv", "full_attention", "conv"],
    }
    config.update(overrides)
    return config


def test_lfm2_feed_forward_width_is_not_the_configs_intermediate_size():
    # The published LFM2.5-1.2B weights are 8,192 wide while the config says 12,288: Liquid's
    # block takes two thirds and rounds up to block_multiple_of. Reading intermediate_size
    # would build a model that cannot load its own checkpoint.
    assert families.lfm2_hidden_dim(_lfm2_config()) == 8192
    # LFM2.5-2.6B turns the auto-adjust off, and then the config's value is the right one.
    assert families.lfm2_hidden_dim(_lfm2_config(block_auto_adjust_ff_dim=False, intermediate_size=10752)) == 10752


def test_lfm2_plan_carries_the_layer_types():
    plan = families.xnnpack_plan(families.family_for(_lfm2_config()), _lfm2_config())
    assert plan.model_class == "lfm2_5_1_2b"
    assert plan.converter == "lfm2"
    assert plan.params["layer_types"] == ["conv", "conv", "full_attention", "conv"]
    assert plan.params["hidden_dim"] == 8192


def test_lfm2_refuses_a_layer_type_list_that_does_not_match():
    with pytest.raises(families.UnsupportedModel):
        families.xnnpack_plan(families.family_for(_lfm2_config()), _lfm2_config(num_hidden_layers=5))


def test_lfm2_is_not_offered_to_backends_without_a_definition():
    family = families.family_for(_lfm2_config())
    assert family.supports("xnnpack")
    for backend in ("vulkan", "qnn", "mtk"):
        assert not family.supports(backend)


def test_lfm2_survives_a_config_that_lost_block_ff_dim():
    # heretic's abliterated LFM2.5-1.2B keeps block_auto_adjust_ff_dim and drops
    # block_ff_dim. intermediate_size holds the same unadjusted 12,288, and the published
    # weights are 8,192 wide, so the rule has to fall back to it rather than fail.
    config = _lfm2_config()
    del config["block_ff_dim"]
    assert families.lfm2_hidden_dim(config) == 8192


def test_a_hybrid_reports_only_the_layers_that_attend():
    # LFM2 keeps a few columns of convolution state instead of a KV cache on most layers,
    # and builds no causal mask there. Counting all 16 as attention layers overstated the
    # 32k export peak enough to refuse a window that fits.
    assert families.attending_layers(_lfm2_config()) == 1  # one full_attention of four
    assert families.attending_layers({"layer_types": ["full_attention"] * 4}) == 4
    # A plain transformer says nothing, and every layer attends.
    assert families.attending_layers({}) is None


def test_the_memory_model_uses_the_attending_count():
    from pipeline import sizing

    arch = families.architecture(_lfm2_config(), 1_200_000_000)
    assert arch.n_layers == 4
    assert arch.attending == 1
    # Masks and KV both scale with the attending layers, not the total.
    assert sizing.causal_mask_bytes(arch, 1024) == 1 * 1024 * 1024
    plain = dataclasses.replace(arch, attention_layers=None)
    assert sizing.causal_mask_bytes(plain, 1024) == 4 * 1024 * 1024
    assert sizing.kv_cache_bytes(arch, 1024) < sizing.kv_cache_bytes(plain, 1024)
