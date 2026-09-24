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

from pipeline import eligibility, families, gate, hub, manifest, mtk_corpus, naming, settings
from pipeline.exporting import (
    RESERVE_BYTES,
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
# backends/mediatek/quantizer/qconfig.py Precision. A16W8 is the one measured against A16W4:
# KL to fp32 0.0068 against 0.75 on LFM2.5-1.2B at 4k (docs/research finding 35).
PRECISIONS = frozenset({"A16W16", "A16W8", "A16W4", "A8W8", "A8W4"})
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
# grows with them), plus WEIGHT_BYTES_PER_PARAM for the weights held throughout. Solved from
# the peak RAM + swap in use each run reported, during "Preparing Model Calibration Inputs":
#
#   Qwen3-0.6B, MT6989, 512, 9 prompts, run 34760462220: 29,592,731,648 B -> 2.37
#   LFM2.5-1.2B, MT6991, 512, 9 prompts, 2026-09-13:     18,383,785,984 B -> 3.76
#
# The factor is not the same across families and one point was never a calibration, so this
# takes the worse of the two and rounds up. Erring low is the expensive direction: the
# trimming loop below only trims until the *estimate* fits, so an estimate that is 1.6x low
# walks the export into an OOM that takes the runner down with it instead of dropping a
# prompt. That is what happened at a 2048 window on the 16.8 GB runner, which this value
# would now refuse up front.
CALIBRATION_OVERHEAD = 3.8
# On top of the prepared calibration inputs, the same filesystem carries the source
# checkpoint, the chunk outputs, the toolchain wheels and the Hub cache. Not the swap file:
# prepare-host.sh writes it before the export starts, so the free space the gate reads has
# already had it taken out. pick_runner, which runs before any of that exists, does subtract
# it. Measured on the 512 runs: a 2.4 GB checkpoint, 1.2 GB of chunks, a few GB of wheels.
DISK_MARGIN_BYTES = 26 * 1024**3
# A window is only given to a runner it fits with room to spare. Exactly filling RAM means
# swapping through the whole calibration, which turns an export into something that times
# out rather than something that fails, and the next tier up is cheaper than finding out.
RUNNER_HEADROOM = 0.85
# model_export_scripts/qwen.py:454-480 keeps the whole checkpoint as a state_dict (16-bit as
# published) and builds every chunk from it in fp32: 2 + 4 bytes per parameter.
WEIGHT_BYTES_PER_PARAM = 6
# third_party/executorch/patches/mediatek-calibration-as-arrays.patch: without it the
# calibration reads every prepared row back as nested Python lists, ~35 min per prompt on
# the hosted runner (docs/research, finding 17).
PATCH_MARKER = 'cal_dataset = cal_dataset.with_format("numpy")'
# third_party/executorch/patches/mediatek-lfm2.patch: lfm2.py calibrates every chunk in one
# streaming pass over the long-context corpus (pipeline/mtk_corpus.py) and keeps no step, so
# the Arrow round trip and the arrays patch do not apply to it.
STREAMING_MARKER = "def calibrate_streaming("
STREAMING_SCRIPTS = frozenset({"lfm2.py"})
# The same patch gives each LFM2 conv layer its own two-position state instead of a
# window-sized K and V it carried only to hide that state in (62% of the cache on the 1.2B),
# plus two padding inputs that keep the runner's pads out of the conv (docs/research, finding
# 36). States go in layer order, one per conv layer, a K and a V per attention layer. MediaTek's
# stock runner cannot load that; the one in the openweights app tells caches from states by
# shape. The runner block records which layout a file has, so a host can tell before loading.
# The script imports the padding inputs from the model file that defines the new layout.
PER_LAYER_STATE_MARKER = "from models.llm_models.modeling_lfm2 import conv_inputs"
PER_LAYER_STATE_SCRIPTS = frozenset({"lfm2.py"})


def state_layout(script: str) -> str:
    return "per-layer" if script in PER_LAYER_STATE_SCRIPTS else "uniform"


# What streaming calibration holds: the checkpoint (2 bytes per parameter), the fp32 chunks (4)
# and every chunk's prepared graph (4); per step the whole cache a few times over (the input,
# the chunks' outputs, their concatenation and its clone) and one block's attention scores.
# Measured on the 1.2B at 512 on the host it was written on: 9.1 GB against 11.9 estimated.
STREAMING_WEIGHT_BYTES_PER_PARAM = 10
STREAMING_CACHE_COPIES = 4
# Lowering peaks above calibration on every export measured so far, at a peak_in_use_bytes of
# 18.4 GB for the 1.2B and 38.8 GB for the 2.6B (the published 512 export reports), 15.7 and
# 14.4 bytes per parameter, and 17.8 GB for the 1.2B at 4k, so it does not grow with the window.
LOWERING_BYTES_PER_PARAM = 16
REPO_ROOT = Path(__file__).resolve().parents[1]
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
        # A .jsonl corpus is already rendered in the model's own chat template.
        *(
            []
            if (dataset or "").endswith(".jsonl")
            else ["--preformatter", f"aot_utils/llm_utils/preformatter_templates/{plan.preformatter}"]
        ),
        "-shapes",
        f"{recipe.prompt_tokens}t{window}c",
        f"1t{window}c",
        "--platform",
        SOC_PLATFORMS[soc],
    ]


def streaming_bytes(config: dict, recipe: settings.MtkRecipe, params: int) -> int:
    """Estimated peak of RAM + swap in use for lfm2.py's streaming calibration or for the
    lowering after it, whichever is higher.

    Nothing is kept per step, so the window enters once, as the cache every step carries and
    one prompt block's attention scores over it, and the corpus size does not enter at all.
    """
    c = families.text_config(config)
    n_heads = int(c["num_attention_heads"])
    head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // n_heads)
    kv_per_token = int(c["num_hidden_layers"]) * 2 * int(c.get("num_key_value_heads") or n_heads) * head_dim * 4
    window = recipe.cache_size
    cache = kv_per_token * window * STREAMING_CACHE_COPIES
    scores = 3 * n_heads * recipe.prompt_tokens * (window + recipe.prompt_tokens) * 4
    calibrate = params * STREAMING_WEIGHT_BYTES_PER_PARAM + cache + scores
    return max(calibrate, params * LOWERING_BYTES_PER_PARAM)


def calibration_bytes(
    config: dict, recipe: settings.MtkRecipe, prompts: int, params: int, *, streaming: bool = False
) -> int:
    """Estimated peak of RAM + swap in use while MediaTek's script prepares calibration inputs,
    the export's peak (docs/research, findings 15, 23, 25).

    model_export_scripts/*.py prepare_model_inputs returns one row per prompt holding the
    fp32 KV cache of every layer, at the full cache size, for the prompt step and each
    generated token up to response_cap. The patched scripts pass writer_batch_size=1, so
    datasets.map keeps **one** of those rows in memory and the finished Arrow file is
    memory mapped, which takes the prompt count out of the peak entirely: it is one prompt's
    steps plus the weights, whatever the corpus size. Before that it was every prompt at
    once, which is what held the window at 512.

    [prompts] no longer changes the answer and is kept so callers and the report read the
    same way; it is asserted against rather than multiplied in. With ``streaming`` (lfm2.py)
    the estimate is streaming_bytes instead.
    """
    if streaming:
        return streaming_bytes(config, recipe, params)
    c = families.text_config(config)
    n_heads = int(c["num_attention_heads"])
    head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // n_heads)
    kv_per_token = int(c["num_hidden_layers"]) * 2 * int(c.get("num_key_value_heads") or n_heads) * head_dim * 4
    steps = 1 + recipe.response_cap
    tensors = steps * kv_per_token * recipe.cache_size
    return int(tensors * CALIBRATION_OVERHEAD) + params * WEIGHT_BYTES_PER_PARAM


def calibration_disk_bytes(config: dict, recipe: settings.MtkRecipe, prompts: int, *, streaming: bool = False) -> int:
    """Bytes the prepared calibration inputs occupy on disk.

    writer_batch_size=1 keeps one prompt in memory, which is what makes a 32k window
    possible, but the rows still all land in the Arrow cache datasets.map writes and are
    read back once per chunk. So the volume did not go away, it moved: every prompt's every
    step, each holding a full fp32 cache at the window. At 32k that is 193 GB for the 1.2B
    and 362 GB for the 2.6B, which is more than the 160 GB a blacksmith-8vcpu runner has.

    Not padded for the checkpoint, the outputs or the swap file: the caller adds those,
    because it knows the sizes and this does not.
    """
    if streaming:
        return 0  # lfm2.py keeps no step, so nothing goes to the Arrow cache
    c = families.text_config(config)
    n_heads = int(c["num_attention_heads"])
    head_dim = int(c.get("head_dim") or int(c["hidden_size"]) // n_heads)
    kv_per_token = int(c["num_hidden_layers"]) * 2 * int(c.get("num_key_value_heads") or n_heads) * head_dim * 4
    return prompts * (1 + recipe.response_cap) * kv_per_token * recipe.cache_size


def pick_runner(
    config: dict,
    recipe: settings.MtkRecipe,
    prompts: int,
    params: int,
    *,
    streaming: bool = False,
) -> settings.RunnerTier | None:
    """The smallest configured runner that can build this window, or None if none can.

    Two limits bind and they bind differently, which is why this is not a constant. Memory
    is one prompt's calibration steps, so it grows with the window and the model. Disk is
    *every* prompt's, because writer_batch_size=1 moved the rest to the Arrow cache. The
    1.2B at 16k wants more memory than the smallest tier has; the 2.6B at 32k wants more
    disk than the middle tier's swap leaves it.

    Memory is asked for twice. A calibration that fits in RAM runs at the speed of the
    forwards; one that fits only with swap reads a multi-gigabyte tensor back through the
    pager on every step, which is the difference between an export that takes an hour and
    one that takes the timeout. So a tier that holds the peak in RAM is preferred, and swap
    is only spent when no tier can, which today is the 2.6B at 32k and nothing else.

    With ``streaming`` (lfm2.py) neither grows much with the window and the time does, so a
    tier is also held to its max_window: a 32k calibration runs a quarter of a million tokens
    through the model twice, and the cores decide whether that fits the job limit.
    """
    need_ram = calibration_bytes(config, recipe, prompts, params, streaming=streaming)
    need_disk = calibration_disk_bytes(config, recipe, prompts, streaming=streaming) + DISK_MARGIN_BYTES

    def carries(tier: settings.RunnerTier, *, with_swap: bool) -> bool:
        swap = tier.swap_gib * 1024**3
        ram = tier.ram_bytes + (swap if with_swap else 0) - RESERVE_BYTES
        fast_enough = not streaming or tier.max_window is None or recipe.cache_size <= tier.max_window
        return need_ram <= ram * RUNNER_HEADROOM and need_disk <= tier.disk_bytes - swap and fast_enough

    for with_swap in (False, True):
        for tier in recipe.runner_tiers:
            if carries(tier, with_swap=with_swap):
                return tier
    return None


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
    precision = os.environ.get("MTK_PRECISION") or recipe.precision
    if precision not in PRECISIONS:
        raise ExportError(f"unknown NeuroPilot precision {precision!r}, expected one of {sorted(PRECISIONS)}")
    recipe = dataclasses.replace(recipe, precision=precision)
    # The prompt graph's batch sets the size of its attention scores, which grow with the
    # window (heads x batch x window per layer), so a smaller batch is a memory lever with a
    # prefill cost; overridable to measure both.
    prompt_tokens = int(os.environ.get("MTK_PROMPT_TOKENS") or recipe.prompt_tokens)
    if prompt_tokens < 1:
        raise ExportError(f"MTK_PROMPT_TOKENS must be positive, got {prompt_tokens}")
    recipe = dataclasses.replace(recipe, prompt_tokens=prompt_tokens)
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
    max_chunks = int(os.environ.get("MTK_MAX_CHUNKS") or recipe.max_chunks)
    plan = families.mtk_plan(family, source.config, max_chunks)
    window = recipe.cache_size
    script = examples_dir / "model_export_scripts" / plan.script
    streaming = plan.script in STREAMING_SCRIPTS
    if streaming and STREAMING_MARKER not in script.read_text(encoding="utf-8"):
        raise ExportError(
            f"{script} lacks the streaming calibration in third_party/executorch/patches/mediatek-lfm2.patch"
        )
    if plan.script in PER_LAYER_STATE_SCRIPTS and PER_LAYER_STATE_MARKER not in script.read_text(encoding="utf-8"):
        raise ExportError(f"{script} lacks the per-layer states in third_party/executorch/patches/mediatek-lfm2.patch")
    if not streaming and PATCH_MARKER not in script.read_text(encoding="utf-8"):
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

    # Calibration memory is one prompt's steps at the full window (calibration_bytes), so
    # the corpus size is free and every prompt is kept. What is not free is the window: a
    # host that cannot hold one prompt's steps cannot build this window at all, and says so
    # rather than being taken down by the OOM killer half an hour in.
    prompts_text = (examples_dir / recipe.calibration).read_text(encoding="utf-8")
    lines = [line for line in prompts_text.splitlines() if line.strip()]
    budget = host_budget(host_info())
    prompts = len(lines) + (recipe.long_samples if streaming else 0)
    params = source.total_params  # eligibility refuses a model without it
    needed = calibration_bytes(source.config, recipe, prompts, params, streaming=streaming)
    if budget is not None and needed > budget:
        what = (
            "the streaming calibration and the lowering"
            if streaming
            else f"one prompt's {1 + recipe.response_cap} steps"
        )
        raise SkipExport(
            f"calibration at a {window}-token cache needs about {needed:,} B for {what}, this host has {budget:,} B"
        )

    # And the same question for disk, because keeping one prompt in memory put every other
    # prompt's steps in the Arrow cache instead. Filling the disk half way through shows up
    # as a write error inside datasets with nothing pointing at the window, so it is worth a
    # sentence up front. The margin covers the checkpoint, the chunk outputs and the swap
    # file, which all sit on the same filesystem.
    disk_needed = calibration_disk_bytes(source.config, recipe, prompts, streaming=streaming) + DISK_MARGIN_BYTES
    work_dir.mkdir(parents=True, exist_ok=True)  # disk_usage needs it to exist
    disk_free = shutil.disk_usage(work_dir).free
    if disk_needed > disk_free:
        raise SkipExport(
            f"calibration at a {window}-token cache writes about {disk_needed:,} B of prepared "
            f"inputs, this host has {disk_free:,} B free on {work_dir}"
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

    corpus = None
    if streaming:
        sources_dir = work_dir / "calibration-sources"
        mtk_corpus.fetch(sources_dir)
        dataset = str((work_dir / f"calibration-{window}.jsonl").resolve())
        summary_path = work_dir / f"calibration-{window}.json"
        build_env = tool_env(tool_python, dict(os.environ))
        build_env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), build_env.get("PYTHONPATH")) if p)
        subprocess.run(
            [
                tool_python, "-m", "pipeline.mtk_corpus",
                "--weights", str(weight_dir), "--sources", str(sources_dir),
                "--window", str(window), "--response-cap", str(recipe.response_cap),
                "--samples", str(recipe.long_samples),
                "--short-prompts", str(examples_dir / recipe.calibration),
                "--out", dataset, "--summary", str(summary_path),
            ],
            check=True, cwd=REPO_ROOT, env=build_env,
        )  # fmt: skip
        corpus = json.loads(summary_path.read_text(encoding="utf-8"))

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
        "state_layout": state_layout(plan.script),
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
            "calibration": (
                {**corpus, "short_prompts": recipe.calibration}
                if corpus
                else {
                    "prompts": recipe.calibration,
                    "prompts_used": prompts,
                    "prompts_available": len(lines),
                    "preformatter": plan.preformatter,
                }
            ),
            "label": f"NeuroPilot {recipe.precision}, {plan.num_chunks} chunks",
            "description": (
                f"ExecuTorch {tools['executorch']} MediaTek LLM export (`examples/mediatek`, "
                f"`{plan.script}`): {recipe.precision} (16-bit activations, "
                f"{recipe.precision.rsplit('W', 1)[-1]}-bit weights) calibrated on "
                + (
                    f"{mtk_corpus.describe(corpus)}, "
                    if corpus
                    else (
                        f"MediaTek's `{recipe.calibration.rsplit('/', 1)[-1]}` prompts ({prompts} of "
                        f"{len(lines)}) in the {plan.preformatter} chat template, "
                    )
                )
                + f"cut into {plan.num_chunks} chunks, with a {recipe.prompt_tokens}-token "
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
