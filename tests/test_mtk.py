import dataclasses
import json
import subprocess

import pytest
from conftest import TOTAL_PARAMS, hf_config, load_json

from pipeline import export_mtk, families, manifest, naming, publish, settings

CFG = settings.load()


def flag(command, name):
    return command[command.index(name) + 1]


def plan(model_id="Qwen/Qwen3-0.6B", config=None):
    config = config or hf_config(model_id)
    return families.mtk_plan(families.family_for(config), config, CFG.mtk.max_chunks)


@pytest.mark.parametrize(("layers", "chunks"), [(28, 4), (24, 4), (36, 4), (30, 3), (26, 2), (1, 1)])
def test_chunks_split_the_layers_evenly(layers, chunks):
    assert families.mtk_chunks(layers, 4) == chunks


def test_qwen_families_use_mediateks_qwen_script_and_their_chat_templates():
    assert plan() == families.MtkPlan("qwen.py", "qwen3.json", 4)
    qwen25 = load_json("qwen2.5-1.5b-instruct.config.json")
    assert families.mtk_plan(families.family_for(qwen25), qwen25, 4) == families.MtkPlan("qwen.py", "qwen.json", 4)


def test_lfm2_uses_lfm2_script_and_chatml_preformatter():
    cfg_1_2b = {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": 16}
    plan_1_2b = families.mtk_plan(families.family_for(cfg_1_2b), cfg_1_2b, CFG.mtk.max_chunks)
    assert plan_1_2b == families.MtkPlan("lfm2.py", "qwen3.json", 4)

    cfg_2_6b = {"architectures": ["Lfm2ForCausalLM"], "model_type": "lfm2", "num_hidden_layers": 30}
    plan_2_6b = families.mtk_plan(families.family_for(cfg_2_6b), cfg_2_6b, CFG.mtk.max_chunks)
    assert plan_2_6b == families.MtkPlan("lfm2.py", "qwen3.json", 3)


def test_rope_scaling_and_other_model_types_are_refused():
    config = hf_config("Qwen/Qwen3-0.6B") | {"rope_scaling": {"rope_type": "yarn", "factor": 4.0}}
    with pytest.raises(families.UnsupportedModel, match="RoPE scaling"):
        plan(config=config)
    with pytest.raises(families.UnsupportedModel, match="model_type"):
        plan(config=hf_config("Qwen/Qwen3-0.6B") | {"model_type": "qwen2"})


def test_export_command_matches_mediateks_shell_scripts(tmp_path):
    command = export_mtk.export_command("/venv/bin/python", plan(), CFG.mtk, "MT6989", tmp_path / "config.json")
    assert command[:3] == ["/venv/bin/python", "model_export_scripts/qwen.py", str(tmp_path / "config.json")]
    assert flag(command, "--precision") == "A16W4"
    assert flag(command, "--num_chunks") == "4"
    assert flag(command, "--dataset") == "aot_utils/llm_utils/prompts/alpaca.txt"
    assert flag(command, "--preformatter") == "aot_utils/llm_utils/preformatter_templates/qwen3.json"
    shapes = command.index("-shapes")
    assert command[shapes + 1 : shapes + 3] == ["128t512c", "1t512c"]
    assert flag(command, "--response_cap") == "9"
    assert flag(command, "--platform") == "DX3"
    assert flag(export_mtk.export_command("p", plan(), CFG.mtk, "MT6991", tmp_path), "--platform") == "DX4"


def test_output_names_follow_the_scripts(tmp_path):
    exp = export_mtk.exp_name(tmp_path / "Qwen3-0.6B", "A16W4", 4)
    assert exp == "Qwen3-0.6B_A16W4_4_chunks"
    assert export_mtk.method_names(exp, CFG.mtk, 3) == [
        "Qwen3-0.6B_A16W4_4_chunks_128t512c_3",
        "Qwen3-0.6B_A16W4_4_chunks_1t512c_3",
    ]


def test_every_chunk_name_reads_as_neuropilot_in_the_app():
    for model_id in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen2.5-1.5B-Instruct"):
        repo = naming.output_repo(model_id, CFG.hub_org, CFG.repo_suffix)
        for soc in CFG.mtk.socs:
            for i in range(4):
                path = f"{naming.mtk_folder(soc)}/{naming.mtk_chunk_file(model_id, 'A16W4', 2048, i, 4)}"
                assert naming.check_app_rules(repo, path, "mtk") == [], path
    assert (
        naming.mtk_chunk_file("Qwen/Qwen3-0.6B", "A16W4", 2048, 0, 4) == "Qwen3-0.6B-neuropilot-a16w4-2k-chunk1of4.pte"
    )


def test_runner_settings_for_qwen3_0_6b():
    runner = export_mtk.runner_settings(hf_config("Qwen/Qwen3-0.6B"), CFG.mtk, 151643, [151645, 151643])
    assert runner["hidden_size"] == 1024
    assert runner["num_head"] == 16
    assert runner["num_layer"] == 28
    assert runner["head_dim"] == 128
    assert runner["rot_emb_base"] == 1000000.0
    assert runner["cache_size"] == 512 and runner["prompt_token_batch_size"] == 128
    assert runner["eos_token"] == 151645 and runner["eos_tokens"] == [151645, 151643]
    assert runner["vocab_size"] == 151936
    assert runner["tokenizer_type"] == "hf" and runner["cache_type"] == "fp32"


def test_a_forced_window_changes_the_shapes(tmp_path):
    recipe = dataclasses.replace(CFG.mtk, cache_size=4096)
    command = export_mtk.export_command("p", plan(), recipe, "MT6991", tmp_path)
    shapes = command.index("-shapes")
    assert command[shapes + 1 : shapes + 3] == ["128t4096c", "1t4096c"]


def test_unknown_chip_and_short_window_are_refused_before_any_download(tmp_path):
    with pytest.raises(export_mtk.ExportError, match="unknown MediaTek chip"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6878", tmp_path, tmp_path, "p", tmp_path)
    with pytest.raises(export_mtk.ExportError, match="below the prompt length"):
        export_mtk.run("Qwen/Qwen3-0.6B", "main", "MT6991", tmp_path, tmp_path, "p", tmp_path, context=64)


def mtk_report(soc="mt6991"):
    chunks = [f"mtk/{soc}/Qwen3-0.6B-neuropilot-a16w4-2k-chunk{i}of4.pte" for i in range(1, 5)]
    embedding = f"mtk/{soc}/Qwen3-0.6B-neuropilot-embedding-fp32.bin"
    return {
        "backend": "mtk",
        "target": soc,
        "target_name": export_mtk.SOC_NAMES[soc.upper()],
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-0.6B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-0.6B", "sha": "c1899de289a0", "license": {"license": "apache-2.0"}},
        "toolchain": {"executorch": "1.4.0"},
        "neuropilot": {
            "name": "NeuroPilot Express SDK",
            "build": "8.0.8-build20250925",
            "mtk_converter": "8.13.0+public",
            "mtk_neuron": "8.2.23",
        },
        "recipe": {"label": "NeuroPilot A16W4, 4 chunks", "description": "recipe."},
        "window": {"context": 2048, "kv_cache_bytes_per_token": None},
        "runner": {"token_embedding_path": embedding.rsplit("/", 1)[-1], "cache_size": 2048},
        "files": [{"path": p, "bytes": 150_000_000, "sha256": "ab"} for p in chunks]
        + [{"path": embedding, "bytes": 622_329_856, "sha256": "cd"}],
        "metadata": {},
        "smoke": {"kind": "structural", "passed": True, "problems": []},
        "run": {},
    }


def test_config_describes_one_model_in_several_files():
    config = manifest.backend_config(mtk_report())
    assert len(config["variants"]) == 1
    variant = config["variants"][0]
    assert variant["files"] == [f"Qwen3-0.6B-neuropilot-a16w4-2k-chunk{i}of4.pte" for i in range(1, 5)]
    assert variant["embedding"] == "Qwen3-0.6B-neuropilot-embedding-fp32.bin"
    assert variant["size_bytes"] == 4 * 150_000_000 + 622_329_856
    # MediaTek's runner flags differ per window (cache size, file names): they live on the variant.
    assert variant["runner"]["cache_size"] == 2048 and variant["context"] == 2048
    assert "runner" not in config
    assert config["neuropilot_sdk"]["build"] == "8.0.8-build20250925"


def test_readme_credits_mediatek_without_claiming_the_sdk_is_included():
    text = manifest.readme("experimentalmachines/Qwen3-0.6B-ExecuTorch", [mtk_report(), mtk_report("mt6989")], [], [])
    assert "MT6991 (Dimensity 9400)" in text and "MT6989 (Dimensity 9300)" in text
    assert "MediaTek NeuroPilot Express SDK 8.0.8-build20250925 (mtk_converter" in text and "MediaTek Inc." in text
    assert "No MediaTek SDK or runtime library is included" in text
    assert "neuropilot-embedding-fp32.bin" in text
    assert "- mtk" in text


def test_publish_refuses_a_tokenizer_inside_a_mediatek_folder(tmp_path):
    folder = tmp_path / "mtk" / "mt6991"
    folder.mkdir(parents=True)
    (folder / "export-report.json").write_text(json.dumps(mtk_report()))
    (folder / "tokenizer.json").write_text("{}")
    with pytest.raises(ValueError, match="repo root"):
        publish.publish_hf(tmp_path, "mtk", "MT6991")


def test_calibration_memory_estimate_covers_the_measured_peak_and_refuses_what_did_not_fit():
    # Qwen3-0.6B: 28 layers x K,V x 8 KV heads x 128 x 4 B = 229,376 B of fp32 cache per token.
    # MediaTek's alpaca.txt holds 9 prompts (8 newlines: its last line has none).
    config = hf_config("Qwen/Qwen3-0.6B")
    params = TOTAL_PARAMS["Qwen/Qwen3-0.6B"]
    weights = 751_632_384 * 6
    at_512 = export_mtk.calibration_bytes(config, CFG.mtk, 9, params)
    assert at_512 == int(9 * 10 * 229_376 * 512 * 2.4) + weights == 29_876_944_896
    # Run 34760462220 peaked at 29,592,731,648 B of RAM + swap in use (finding 25).
    assert at_512 >= 29_592_731_648
    # 16.8 GB RAM + 24 GB swap minus the reserve: 512 fits with all 9 prompts; 2048 does not
    # fit even with mtk.min_calibration_prompts (run 1 took the runner down at 2048).
    budget = 16_765_378_560 + 25_769_799_680 - 1_000_000_000
    at_2k = dataclasses.replace(CFG.mtk, cache_size=2048)
    assert at_512 < budget
    assert export_mtk.calibration_bytes(config, at_2k, CFG.mtk.min_calibration_prompts, params) == 49_606_950_912
    assert 49_606_950_912 > budget


def test_the_calibration_patch_adds_what_the_export_checks_for():
    patch = settings.ROOT / "third_party/executorch/patches/mediatek-calibration-as-arrays.patch"
    added = [line[1:].strip() for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("+ ")]
    assert export_mtk.PATCH_MARKER in added
    # One family script per patch target: every mapped script must be covered.
    targets = {line.split("/")[-1] for line in patch.read_text(encoding="utf-8").splitlines() if line.startswith("+++")}
    assert {plan().script} <= targets


def test_the_export_script_starts_with_mtk_neurons_lib_on_the_loader_path(monkeypatch):
    lib = "/w/mtk-venv/lib/python3.10/site-packages/mtk_neuron/lib"

    def fake_run(command, **kwargs):
        assert command[1] == "-c" and "mtk_neuron" in command[2]
        return subprocess.CompletedProcess(command, 0, stdout=lib + "\n", stderr="")

    monkeypatch.setattr(export_mtk.subprocess, "run", fake_run)
    env = export_mtk.tool_env("/w/mtk-venv/bin/python", {"LD_LIBRARY_PATH": "/opt/x", "HOME": "/h"})
    assert env["LD_LIBRARY_PATH"] == f"{lib}:/opt/x"
    assert env["HOME"] == "/h" and env["PYTHONUNBUFFERED"] == "1"
    assert export_mtk.tool_env("p", {})["LD_LIBRARY_PATH"] == lib
