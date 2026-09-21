"""Export one HF model to MediaTek NeuroPilot .pte chunks for one Dimensity chip.

Drives ExecuTorch 1.4.0's examples/mediatek LLM scripts, which are not in the executorch
wheel (the workflow fetches them from the ExecuTorch source at EXECUTORCH_COMMIT), in a
separate Python 3.10 environment holding MediaTek's mtk_converter and mtk_neuron from the
NeuroPilot Express SDK. The script quantizes the model (A16W4), cuts it into chunks of equal
layer counts, and compiles a prompt graph and a one-token generation graph per chunk for the
chip. Produces, under ``out_dir``:

    tokenizer.json | tokenizer.model, LICENSE…, NOTICE   (repo root, shared with other backends)
    mtk/<soc>/<name>-neuropilot-a16w4-<window>-chunk<i>of<n>.pte
    mtk/<soc>/<name>-neuropilot-embedding-fp32.bin       (token embedding table the runner reads)
    mtk/<soc>/config.json                                (includes MediaTek runner settings)
    mtk/<soc>/export-report-<window>.json

These run on MediaTek's LLM runner (examples/mediatek/executor_runner), not on ExecuTorch's
generic TextLLMRunner. There is no host runtime for NeuroPilot binaries, so the check after
export is structural: every chunk loads, carries both graphs, and delegates to NeuropilotBackend.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from pipeline import eligibility, families, gate, hub, manifest, naming, settings
from pipeline.exporting import (
    ExportError,
    MemorySampler,
    SkipExport,
    children_peak_rss,
    contains,
    copy_side_files,
    host_budget,
    host_info,
    report_file,
    sha256,
    source_report,
)

BACKEND = "mtk"
DELEGATE = b"NeuropilotBackend"
# examples/mediatek/model_export_scripts/*.py --platform
SOC_PLATFORMS = {"MT6989": "DX3", "MT6991": "DX4"}
SOC_NAMES = {"MT6989": "Dimensity 9300", "MT6991": "Dimensity 9400"}
# IO types of the graphs these scripts export, as ExecuTorch's run_qwen2_sample.sh and
# run_qwen3_sample.sh pass them to the runner.
RUNNER_IO_TYPES = {
    "input_type": "fp32",
    "output_type": "fp32",
    "cache_type": "fp32",
    "mask_type": "fp32",
    "rot_emb_type": "fp32",
}
# calibration_bytes: the calibration tensors times this (their Arrow copy and the rest that
# grows with them), plus WEIGHT_BYTES_PER_PARAM for the weights held throughout. Fitted to
# run 34760462220 (Qwen3-0.6B, MT6989, 512, 9 prompts): 29,592,731,648 B of RAM + swap in
# use at its peak, during "Preparing Model Calibration Inputs" (docs/research, finding 25).
CALIBRATION_OVERHEAD = 2.4
# model_export_scripts/qwen.py:454-480 keeps the whole checkpoint as a state_dict (16-bit as
# published) and builds every chunk from it in fp32: 2 + 4 bytes per parameter.
WEIGHT_BYTES_PER_PARAM = 6
# third_party/executorch/patches/mediatek-calibration-as-arrays.patch: without it the
# calibration reads every prepared row back as nested Python lists, ~35 min per prompt on
# the hosted runner (docs/research, finding 17).
PATCH_MARKER = 'cal_dataset = cal_dataset.with_format("numpy")'
TOOL_PACKAGES = ("executorch", "torch", "torchao", "transformers", "mtk_converter", "mtk-neuron")

_LOAD_PROGRAM = """
import json, sys
from executorch.runtime import Runtime
print(json.dumps(sorted(Runtime.get().load_program(sys.argv[1]).method_names)))
"""
_VERSIONS = """
import json, re, sys
from importlib import metadata
norm = lambda name: re.sub(r"[-_.]+", "-", name).lower()
installed = {norm(d.metadata["Name"]): d.version for d in metadata.distributions()}
print(json.dumps({name: installed.get(norm(name)) for name in sys.argv[1:]}))
"""


def export_command(
    tool_python: str,
    plan: families.MtkPlan,
    recipe: settings.MtkRecipe,
    soc: str,
    config_path: Path,
    dataset: str | None = None,
) -> list[str]:
    window = recipe.cache_size
    return [
        tool_python,
        f"model_export_scripts/{plan.script}",
        str(config_path),
        "--precision",
        recipe.precision,
        "--num_chunks",
        str(plan.num_chunks),
        "--dataset",
        dataset or recipe.calibration,
        "--response_cap",
        str(recipe.response_cap),
        "--preformatter",
        f"aot_utils/llm_utils/preformatter_templates/{plan.preformatter}",
        "-shapes",
        f"{recipe.prompt_tokens}t{window}c",
        f"1t{window}c",
        "--platform",
        SOC_PLATFORMS[soc],
    ]


def calibration_bytes(config: dict, recipe: settings.MtkRecipe, prompts: int, params: int) -> int:
    """Estimated peak of RAM + swap in use while MediaTek's script prepares calibration inputs,
    the export's peak (docs/research, findings 15, 23, 25).

    For every prompt, model_export_scripts/*.py prepare_model_inputs keeps the fp32 KV cache
    of every layer at the full cache size for the prompt step and each generated token (up
    to response_cap), and datasets.map then holds them all as Arrow rows before writing,
    while the weights stay loaded. One measured point (see CALIBRATION_OVERHEAD) fixes the
    factor, so how the peak splits between the two terms is an assumption.
    """
    c = families.text_config(config)
    n_heads = int(c["num_attention_heads"])
    head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // n_heads)
    kv_per_token = int(c["num_hidden_layers"]) * 2 * int(c.get("num_key_value_heads") or n_heads) * head_dim * 4
    steps = prompts * (1 + recipe.response_cap)
    tensors = steps * kv_per_token * recipe.cache_size
    return int(tensors * CALIBRATION_OVERHEAD) + params * WEIGHT_BYTES_PER_PARAM


def exp_name(weight_dir: Path, precision: str, chunks: int) -> str:
    """The name the scripts give the output folder and files (utils.get_exp_name)."""
    return f"{weight_dir.name}_{precision}_{chunks}_chunks"


def method_names(exp: str, recipe: settings.MtkRecipe, chunk: int) -> list[str]:
    window = recipe.cache_size
    return sorted(f"{exp}_{shape}_{chunk}" for shape in (f"{recipe.prompt_tokens}t{window}c", f"1t{window}c"))


def runner_settings(config: dict, recipe: settings.MtkRecipe, bos: int | None, eos: list[int]) -> dict:
    """Flags for MediaTek's mtk_llama_executor_runner (file names are added by the caller)."""
    c = families.text_config(config)
    n_heads = int(c["num_attention_heads"])
    eos_token = c.get("eos_token_id")
    if isinstance(eos_token, list):
        eos_token = eos_token[0] if eos_token else None
    return {
        "prompt_token_batch_size": recipe.prompt_tokens,
        "cache_size": recipe.cache_size,
        "hidden_size": int(c["hidden_size"]),
        "num_head": n_heads,
        "num_layer": int(c["num_hidden_layers"]),
        "head_dim": int(c.get("head_dim") or int(c["hidden_size"]) // n_heads),
        "max_token_length": int(c["max_position_embeddings"]),
        "rot_emb_base": families.rope_theta(c),
        **RUNNER_IO_TYPES,
        "vocab_size": int(c["vocab_size"]),
        "bos_token": bos if bos is not None else c.get("bos_token_id"),
        "eos_token": eos_token if eos_token is not None else (eos[0] if eos else None),
        "eos_tokens": eos,
        "tokenizer_type": "hf",
    }


def structural_check(tool_python: str, chunks: list[Path], exp: str, recipe: settings.MtkRecipe) -> dict:
    problems = []
    methods = {}
    for i, pte in enumerate(chunks):
        result = subprocess.run([tool_python, "-c", _LOAD_PROGRAM, str(pte)], capture_output=True, text=True)
        if result.returncode != 0:
            problems.append(f"{pte.name} did not load: {result.stderr.strip()[-500:]}")
            continue
        found = json.loads(result.stdout.strip().splitlines()[-1])
        methods[pte.name] = found
        wanted = method_names(exp, recipe, i)
        if found != wanted:
            problems.append(f"{pte.name} has methods {found}, expected {wanted}")
        if not contains(pte, DELEGATE):
            problems.append(f"{pte.name} has no NeuropilotBackend delegate")
    return {"kind": "structural", "passed": not problems, "problems": problems, "methods": methods}


_NEURON_LIB = (
    "import mtk_neuron, os; print(os.path.join(os.path.dirname(os.path.realpath(mtk_neuron.__file__)), 'lib'))"
)


def tool_env(tool_python: str, base: dict[str, str]) -> dict[str, str]:
    """Environment for MediaTek's export script: mtk_neuron's own lib/ on the loader path.

    mtk_neuron 8.2.23 ctypes-loads lib/libextract_shared.so, which needs libc++.so.1 by name
    and has no RPATH; the wheel ships that libc++ in the same lib/ folder, where the loader
    only looks if LD_LIBRARY_PATH says so when the process starts (docs/research, finding 21).
    """
    result = subprocess.run([tool_python, "-c", _NEURON_LIB], capture_output=True, text=True)
    if result.returncode != 0:
        raise ExportError(f"cannot locate mtk_neuron in the MediaTek tool environment:\n{result.stderr[-2000:]}")
    neuron_lib = result.stdout.strip().splitlines()[-1]
    paths = [neuron_lib] + [p for p in base.get("LD_LIBRARY_PATH", "").split(":") if p]
    return {**base, "LD_LIBRARY_PATH": ":".join(paths), "PYTHONUNBUFFERED": "1"}


def tool_versions(tool_python: str) -> dict:
    result = subprocess.run([tool_python, "-c", _VERSIONS, *TOOL_PACKAGES], capture_output=True, text=True)
    if result.returncode != 0:
        raise ExportError(f"MediaTek tool environment is broken:\n{result.stderr[-2000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def run(
    model_id: str,
    revision: str,
    soc: str,
    out_dir: Path,
    work_dir: Path,
    tool_python: str,
    examples_dir: Path,
    keep_work: bool = False,
    context: int | None = None,
) -> dict:
    cfg = settings.load()
    recipe = cfg.mtk
    if soc not in SOC_PLATFORMS:
        raise ExportError(f"unknown MediaTek chip {soc!r}; ExecuTorch 1.4.0's scripts know {sorted(SOC_PLATFORMS)}")
    if context is not None:
        if context < recipe.prompt_tokens:
            raise ExportError(f"--context {context} is below the prompt length {recipe.prompt_tokens}")
        recipe = dataclasses.replace(recipe, cache_size=context)
    if not (examples_dir / "model_export_scripts").is_dir():
        raise ExportError(f"{examples_dir} is not ExecuTorch's examples/mediatek")
    source = hub.fetch(model_id, revision)
    verdict = eligibility.evaluate(source, cfg)
    if verdict.reasons:
        raise ExportError(f"{model_id} is not eligible: {'; '.join(verdict.reasons)}")
    if verdict.backends[BACKEND] is not None:
        raise ExportError(f"{model_id} cannot be exported to {BACKEND}: {verdict.backends[BACKEND]}")
    family = families.family_for(source.config)
    plan = families.mtk_plan(family, source.config, recipe.max_chunks)
    window = recipe.cache_size
    script = examples_dir / "model_export_scripts" / plan.script
    if PATCH_MARKER not in script.read_text(encoding="utf-8"):
        raise ExportError(f"{script} lacks third_party/executorch/patches/mediatek-calibration-as-arrays.patch")

    output_repo = naming.output_repo(model_id, cfg.hub_org, cfg.repo_suffix)
    folder = naming.mtk_folder(soc)
    chunk_names = [
        naming.mtk_chunk_file(model_id, recipe.precision, window, i, plan.num_chunks) for i in range(plan.num_chunks)
    ]
    embedding_name = naming.mtk_embedding_file(model_id)
    for name in chunk_names:
        problems = naming.check_app_rules(output_repo, f"{folder}/{name}", BACKEND)
        if problems:
            raise ExportError("; ".join(problems))

    # Calibration memory grows with the window (calibration_bytes): larger windows keep
    # fewer of MediaTek's prompts, down to mtk.min_calibration_prompts; past that the window
    # is skipped on this host rather than calibrated on too little.
    prompts_text = (examples_dir / recipe.calibration).read_text(encoding="utf-8")
    lines = [line for line in prompts_text.splitlines() if line.strip()]
    budget = host_budget(host_info())
    prompts = len(lines)
    floor = min(recipe.min_calibration_prompts, prompts)
    params = source.total_params  # eligibility refuses a model without it

    def needs(n: int) -> int:
        return calibration_bytes(source.config, recipe, n, params)

    while budget is not None and prompts > floor and needs(prompts) > budget:
        prompts -= 1
    needed = needs(prompts)
    if budget is not None and needed > budget:
        raise SkipExport(
            f"calibration at a {window}-token cache needs about {needed:,} B even with {prompts} prompts "
            f"({1 + recipe.response_cap} steps each), this host has {budget:,} B"
        )
    dataset = recipe.calibration
    if prompts < len(lines):
        work_dir.mkdir(parents=True, exist_ok=True)
        trimmed = work_dir / "calibration-prompts.txt"
        trimmed.write_text("\n".join(lines[:prompts]) + "\n", encoding="utf-8")
        dataset = str(trimmed.resolve())

    tools = tool_versions(tool_python)
    for package, pinned in (("mtk_converter", "MTK_CONVERTER_VERSION"), ("mtk-neuron", "MTK_NEURON_VERSION")):
        expected = settings.read_env_file(settings.CONFIG_DIR / "versions.env")[pinned]
        if not (tools.get(package) or "").startswith(expected):
            raise ExportError(f"{package} is {tools.get(package)}, config/versions.env pins {expected}")

    backend_dir = out_dir / folder
    backend_dir.mkdir(parents=True, exist_ok=True)
    # The scripts name their outputs after the weight directory.
    weight_dir = work_dir / "weights" / naming.source_name(model_id)
    print(f"==> {model_id}@{source.sha[:12]}: {plan.script}, {plan.num_chunks} chunks, {soc}, window {window}")

    started = time.time()
    hub.download(source, weight_dir)
    tokenizer, licenses = copy_side_files(source, weight_dir, out_dir)
    bos, eos = hub.special_token_ids(source)

    command = export_command(tool_python, plan, recipe, soc, weight_dir / "config.json", dataset)
    print("==> " + " ".join(command))
    export_started = time.time()
    env = tool_env(tool_python, dict(os.environ))
    with MemorySampler() as memory:
        subprocess.run(command, check=True, cwd=examples_dir, env=env)
    export_seconds = time.time() - export_started

    exp = exp_name(weight_dir, recipe.precision, plan.num_chunks)
    built = [examples_dir / "pte" / exp / f"{exp}_{i}.pte" for i in range(plan.num_chunks)]
    missing = [p.name for p in built if not p.is_file()]
    if missing:
        raise ExportError(f"the export wrote no {missing} in {examples_dir / 'pte' / exp}")
    chunks = []
    for path, name in zip(built, chunk_names, strict=True):
        shutil.move(str(path), backend_dir / name)
        chunks.append(backend_dir / name)
    embedding = backend_dir / embedding_name
    shutil.move(str(weight_dir / f"embedding_{weight_dir.name}_fp32.bin"), embedding)
    c = families.text_config(source.config)
    expected_bytes = int(c["vocab_size"]) * int(c["hidden_size"]) * 4
    if embedding.stat().st_size != expected_bytes:
        raise ExportError(f"embedding table is {embedding.stat().st_size:,} B, expected {expected_bytes:,} B")

    check = structural_check(tool_python, chunks, exp, recipe)
    runner = {
        **runner_settings(source.config, recipe, bos, eos),
        "tokenizer_path": tokenizer,
        "token_embedding_path": embedding_name,
        "model_package_paths": chunk_names,
    }
    sdk_versions = settings.read_env_file(settings.CONFIG_DIR / "versions.env")
    sdk = {
        "name": "NeuroPilot Express SDK",
        "build": sdk_versions["NEUROPILOT_SDK_BUILD"],
        "mtk_converter": tools.get("mtk_converter"),
        "mtk_neuron": tools.get("mtk-neuron"),
    }
    files = [
        {"path": f"{folder}/{p.name}", "bytes": p.stat().st_size, "sha256": sha256(p)} for p in [*chunks, embedding]
    ]
    report = {
        "backend": BACKEND,
        "target": soc.lower(),
        "target_name": SOC_NAMES[soc],
        "tokenizer": tokenizer,
        "output_repo": output_repo,
        "source": source_report(source, verdict.family, verdict.variant, licenses),
        "toolchain": {k: v for k, v in tools.items() if k not in ("mtk_converter", "mtk-neuron")},
        "neuropilot": sdk,
        "recipe": {
            "script": plan.script,
            "precision": recipe.precision,
            "num_chunks": plan.num_chunks,
            "prompt_tokens": recipe.prompt_tokens,
            "cache_size": window,
            "calibration": {
                "prompts": recipe.calibration,
                "prompts_used": prompts,
                "prompts_available": len(lines),
                "preformatter": plan.preformatter,
            },
            "label": f"NeuroPilot {recipe.precision}, {plan.num_chunks} chunks",
            "description": (
                f"ExecuTorch {tools['executorch']} MediaTek LLM export (`examples/mediatek`, "
                f"`{plan.script}`): {recipe.precision} (16-bit activations, "
                f"{recipe.precision.rsplit('W', 1)[-1]}-bit weights) calibrated on MediaTek's "
                f"`{recipe.calibration.rsplit('/', 1)[-1]}` prompts ({prompts} of {len(lines)}) in the "
                f"{plan.preformatter} chat "
                f"template, cut into {plan.num_chunks} chunks, with a {recipe.prompt_tokens}-token "
                f"prompt graph and a one-token generation graph over a {window}-token cache, "
                f"compiled with MediaTek NeuroPilot Express SDK (mtk_converter {sdk['mtk_converter']}, "
                f"mtk_neuron {sdk['mtk_neuron']}) for {soc} ({SOC_NAMES[soc]})."
            ),
        },
        "window": {
            "context": window,
            "reason": (
                "forced with --context (static NPU graphs)"
                if context is not None
                else "fixed by mtk.cache_size (static NPU graphs)"
            ),
            "kv_cache_bytes_per_token": None,
        },
        "runner": runner,
        "files": files,
        "host": {
            **host_info(),
            "peak_rss_export_bytes": children_peak_rss(),
            **memory.result(),
            "export_seconds": round(export_seconds, 1),
            "total_seconds": round(time.time() - started, 1),
        },
        "metadata": {},
        "smoke": check,
        "measuring_gate": gate.measuring_gate(check.get("stats")),
        "run": manifest.run_info(),
    }
    (backend_dir / "config.json").write_text(
        json.dumps(manifest.backend_config(report), indent=2) + "\n", encoding="utf-8"
    )
    (backend_dir / report_file(window)).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
        shutil.rmtree(examples_dir / "pte" / exp, ignore_errors=True)
    if not check["passed"]:
        raise ExportError("structural check failed: " + "; ".join(check["problems"]))
    return report
