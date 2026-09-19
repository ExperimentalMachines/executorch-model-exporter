import pytest
from conftest import CONFIG_FILES, make_source

from pipeline import eligibility, naming, settings


@pytest.mark.parametrize(
    "name, family",
    [
        ("Qwen3-1.7B-ExecuTorch-Qwen3-1.7B-8da4w-8k.pte", "qwen3"),
        ("Qwen2.5-1.5B-Instruct-ExecuTorch", "qwen25"),
        ("Llama-3.2-1B-Instruct-ExecuTorch", "llama32"),
        ("SmolLM2-360M-Instruct", "smollm2"),
        ("gemma-3-1b-it", "gemma3"),
        ("LFM2.5-VL-1.6B", "lfm25"),
        # The app has had Qwen35Template since 2026-09; it is a family, not an exclusion.
        ("Qwen3.5-2B", "qwen35"),
        ("Qwen2.5-VL-3B-Instruct", None),
        ("Qwen2.5-Coder-1.5B-Instruct", None),
        ("Llama-Guard-3-1B", None),
        ("Qwen2-1.5B-Instruct", None),
    ],
)
def test_app_family(name, family):
    assert naming.app_family(name) == family


@pytest.mark.parametrize(
    "text, backend",
    [
        ("experimentalmachines/Qwen3-1.7B-ExecuTorch/xnnpack/Qwen3-1.7B-8da4w-8k.pte", "xnnpack"),
        ("experimentalmachines/Qwen3-1.7B-ExecuTorch/qnn/sm8650/model.pte", "qnn"),
        ("experimentalmachines/Qwen3-1.7B-ExecuTorch/mtk/mt6991/model.pte", "neuropilot"),
        # The trap the plan's layout avoids: "xnnpack" in the repo name wins.
        ("someone/Qwen3-1.7B-ExecuTorch-XNNPACK/qnn/sm8650/model.pte", "xnnpack"),
        ("someone/Qwen3-1.7B/model.pte", "unknown"),
    ],
)
def test_app_backend(text, backend):
    assert naming.app_backend(text) == backend


def test_app_model_name_matches_the_kotlin_examples():
    # Examples from ExecuTorchFileName.modelNameFor's KDoc.
    assert (
        naming.app_model_name("larryliu0820/Qwen3-1.7B-INT8-INT4-ExecuTorch-XNNPACK", "model.pte")
        == "Qwen3-1.7B-INT8-INT4-ExecuTorch-XNNPACK.pte"
    )
    assert naming.app_model_name("someone/two-sizes", "3b/xnnpack/model.pte") == "two-sizes-3b-xnnpack.pte"
    assert (
        naming.app_model_name("experimentalmachines/Qwen3-1.7B-ExecuTorch", "xnnpack/Qwen3-1.7B-8da4w-8k.pte")
        == "Qwen3-1.7B-ExecuTorch-Qwen3-1.7B-8da4w-8k.pte"
    )


@pytest.mark.parametrize(
    "name, hint, billions",
    [
        ("Qwen3-1.7B-ExecuTorch", "1.7B", 1.7),
        ("Llama-3.2-1B-Instruct-ExecuTorch", "1B", 1.0),
        ("SmolLM2-135M-Instruct", "135M", 0.135),
        ("Qwen3-4B-Instruct-2507", "4B", 4.0),
        ("gemma-3-1b-it", "1B", 1.0),
        ("no-size-here", None, None),
    ],
)
def test_size_hint(name, hint, billions):
    assert naming.app_size_hint(name) == hint
    assert naming.nominal_billions(name) == (pytest.approx(billions) if billions else None)


def test_every_generated_name_passes_the_app_rules():
    cfg = settings.load()
    checked = 0
    for model_id in CONFIG_FILES:
        verdict = eligibility.evaluate(make_source(model_id), cfg)
        if not verdict.eligible:
            continue
        repo = naming.output_repo(model_id, cfg.hub_org, cfg.repo_suffix)
        for context in cfg.context_tiers:
            path = f"xnnpack/{naming.xnnpack_file(model_id, cfg.xnnpack.qmode, context)}"
            assert naming.check_app_rules(repo, path, "xnnpack") == [], (repo, path)
        checked += 1
    assert checked >= 5


def test_app_rule_violations_are_reported():
    problems = naming.check_app_rules("someone/Qwen3-1.7B-XNNPACK", "qnn/sm8650/model.pte", "qnn")
    assert any("contains 'xnnpack'" in p for p in problems)
    assert any("reads as backend 'xnnpack'" in p for p in problems)
    # A name the app genuinely cannot place: "vl" is excluded there and here, so a vision
    # export must not pass the rules even though its family token would otherwise match.
    assert naming.check_app_rules("someone/Qwen3.5-VL-2B-ExecuTorch", "xnnpack/a.pte", "xnnpack")


def test_names_with_several_or_no_sizes_are_refused_by_name():
    cfg = settings.load()
    assert naming.size_hints("Qwen2.5-1M-1.5B") == ["1.5B", "1M"]
    assert any("several sizes" in r for r in eligibility.name_reasons("Qwen/Qwen2.5-1M-1.5B", cfg))
    assert any("has none" in r for r in eligibility.name_reasons("Qwen/Qwen3-Next", cfg))
    assert eligibility.name_reasons("Qwen/Qwen3-1.7B", cfg) == []


def test_window_label():
    assert naming.window_label(32768) == "32k"
    assert naming.window_label(8192) == "8k"
    assert naming.window_label(3000) == "3000"
