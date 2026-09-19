import pytest
from conftest import hf_config, make_source

from pipeline import eligibility, settings

CFG = settings.load()


def evaluate(model_id, **overrides):
    return eligibility.evaluate(make_source(model_id, **overrides), CFG)


def test_qwen3_goes_to_every_backend():
    verdict = evaluate("Qwen/Qwen3-1.7B")
    assert verdict.eligible
    assert verdict.reasons == []
    assert verdict.export_backends == ["xnnpack", "vulkan", "qnn", "mtk"]
    assert verdict.variant == "instruct"


def test_qnn_needs_an_entry_in_executorchs_qualcomm_registry():
    # Same architecture, but ExecuTorch 1.4.0's Qualcomm scripts only list Qwen/Qwen3-1.7B.
    # The MediaTek scripts build any Qwen3 from its config.json.
    verdict = evaluate("Qwen/Qwen3-1.7B-Base", config=hf_config("Qwen/Qwen3-1.7B"))
    assert verdict.export_backends == ["xnnpack", "vulkan", "mtk"]
    assert "no entry" in verdict.backends["qnn"]


def test_mediatek_waits_on_validation_for_llama():
    verdict = evaluate("meta-llama/Llama-3.2-1B-Instruct")
    assert "mtk" not in verdict.export_backends
    assert "rope_type llama3" in verdict.backends["mtk"]


def test_the_4b_class_is_in_by_name_even_though_it_has_4_02b_parameters():
    verdict = evaluate("Qwen/Qwen3-4B")
    assert verdict.eligible, verdict.reasons


def test_named_size_above_the_cap():
    verdict = evaluate("Qwen/Qwen3-8B", config=hf_config("Qwen/Qwen3-4B"), total_params=3_000_000_000)
    assert any("named size 8B" in r for r in verdict.reasons)


def test_real_count_cap_applies_without_a_name():
    verdict = evaluate("someone/Qwen3-mystery", config=hf_config("Qwen/Qwen3-4B"), total_params=7_000_000_000)
    assert any("not below" in r for r in verdict.reasons)


def test_moe_is_refused_by_name_of_the_problem():
    verdict = evaluate("Qwen/Qwen3-30B-A3B")
    assert "mixture-of-experts checkpoint" in verdict.reasons
    assert not verdict.eligible


def test_prequantized_and_non_chat_repos():
    fp8 = evaluate("Qwen/Qwen3-1.7B-FP8", config=hf_config("Qwen/Qwen3-1.7B"))
    assert any("excluded marker" in r for r in fp8.reasons)
    embedding = evaluate("Qwen/Qwen3-Embedding-0.6B", config=hf_config("Qwen/Qwen3-0.6B"))
    assert any("excluded marker" in r for r in embedding.reasons)
    ranker = evaluate("Qwen/Qwen3-1.7B", pipeline_tag="text-ranking")
    assert any("pipeline tag" in r for r in ranker.reasons)


def test_names_the_app_would_refuse():
    coder = evaluate("Qwen/Qwen2.5-Coder-1.5B-Instruct", config=hf_config("Qwen/Qwen2.5-1.5B-Instruct"))
    assert any("no chat template" in r for r in coder.reasons)
    qwen2 = evaluate("Qwen/Qwen2-1.5B-Instruct", config=hf_config("Qwen/Qwen2.5-1.5B-Instruct"))
    assert any("no chat template" in r for r in qwen2.reasons)


def test_name_and_architecture_must_agree():
    mislabelled = evaluate("someone/SmolLM2-1.7B", config=hf_config("Qwen/Qwen3-1.7B"))
    assert any("architecture is 'qwen3'" in r for r in mislabelled.reasons)


def test_llama_family_covers_llama32_and_smollm2():
    assert evaluate("meta-llama/Llama-3.2-1B-Instruct").eligible
    assert evaluate("HuggingFaceTB/SmolLM2-360M-Instruct").eligible


def test_gemma3_goes_to_qnn_only():
    verdict = evaluate("google/gemma-3-1b-it")
    assert verdict.reasons == []
    assert verdict.export_backends == ["qnn"]
    assert "export_llm" in verdict.backends["xnnpack"]


def test_gated_repo_without_access_says_what_to_do():
    verdict = evaluate("google/gemma-3-1b-it", config={}, access_error="gated repo and the HF_TOKEN account ...")
    assert verdict.reasons[0].startswith("gated repo")
    assert not verdict.eligible


def test_a_later_generation_on_the_same_architecture_class_is_refused():
    # "Qwen3.8" normalises onto "qwen3", so the app's own matcher would accept it as Qwen3.
    verdict = evaluate("Qwen/Qwen3.8-1B", config=hf_config("Qwen/Qwen3-0.6B"))
    assert any("not named like a qwen3 release" in r for r in verdict.reasons)
    gemma3n = evaluate("google/gemma-3n-E2B-it", config=hf_config("google/gemma-3-1b-it"))
    assert any("not named like a gemma3 release" in r for r in gemma3n.reasons)


def test_name_reasons_need_no_network():
    assert eligibility.name_reasons("Qwen/Qwen3-1.7B", CFG) == []
    assert eligibility.name_reasons("Qwen/Qwen3-1.7B-GGUF", CFG)
    assert eligibility.name_reasons("Qwen/Qwen3-32B", CFG)
    assert eligibility.name_reasons("Qwen/Qwen2.5-VL-3B-Instruct", CFG)


def test_variant():
    assert eligibility.variant("Qwen/Qwen3-1.7B", "qwen3") == "instruct"
    assert eligibility.variant("Qwen/Qwen3-1.7B-Base", "qwen3") == "base"
    assert eligibility.variant("Qwen/Qwen2.5-1.5B", "qwen2_5") == "base"
    assert eligibility.variant("Qwen/Qwen2.5-1.5B-Instruct", "qwen2_5") == "instruct"
    assert eligibility.variant("google/gemma-3-1b-it", "gemma3") == "instruct"
    assert eligibility.variant("meta-llama/Llama-3.2-1B", "llama") == "base"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        # LFM2.5 is uneven: the 1.2B names its chat model, the 2.6B does not.
        ("LiquidAI/LFM2.5-1.2B-Instruct", "instruct"),
        ("LiquidAI/LFM2.5-1.2B-Base", "base"),
        ("LiquidAI/LFM2.5-2.6B", "instruct"),
        ("LiquidAI/LFM2.5-2.6B-Base", "base"),
        # An abliterated copy keeps whatever it was abliterated from.
        ("experimentalmachines/LFM2.5-2.6B-heretic", "instruct"),
        ("experimentalmachines/LFM2.5-1.2B-Instruct-heretic", "instruct"),
    ],
)
def test_lfm2_chat_models_are_not_read_as_base(model_id, expected):
    """LFM2.5-2.6B is the instruct model; LFM2.5-2.6B-Base is the base.

    Reading the bare name as a base model made the smoke test prompt a chat model with a
    completion string and recorded "base" in every published report for it.
    """
    assert eligibility.variant(model_id, "lfm2") == expected
