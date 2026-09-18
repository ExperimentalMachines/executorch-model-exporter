"""Export one HF model to an XNNPACK (CPU) or Vulkan (GPU) .pte with ExecuTorch's export_llm.

Both backends share the recipe (8da4w linears, int8 embeddings, fp32 KV cache): the Vulkan
delegate runs torchao's Int8DynamicActivationIntxWeight linears (docs/source/backends/vulkan/
vulkan-quantization.md), so only ``backend`` in the export_llm config differs. An XNNPACK
file is smoke-tested with the wheel's TextLLMRunner; the runner has no Vulkan kernels and
the host no GPU, so a Vulkan file gets the structural check (loads, delegates to
VulkanBackend).

Produces, under ``out_dir`` (the layout of the published HF repo):

    tokenizer.json | tokenizer.model
    LICENSE…, NOTICE
    xnnpack/<name>-8da4w-<window>.pte
    xnnpack/config.json
    xnnpack/export-report-<window>.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

from pipeline import convert, eligibility, families, hub, manifest, naming, settings, sizing, smoke
from pipeline.exporting import (
    ExportError,
    MemorySampler,
    SkipExport,
    children_peak_rss,
    copy_side_files,
    host_budget,
    host_info,
    report_file,
    run_tool,
    self_peak_rss,
    sha256,
    source_report,
    toolchain,
)

BACKEND = "xnnpack"
BACKEND_CONFIG = {
    "xnnpack": {"xnnpack": {"enabled": True, "extended_ops": True}},
    "vulkan": {"vulkan": {"enabled": True}},
}
DELEGATE_ID = {"xnnpack": "XnnpackBackend", "vulkan": "VulkanBackend"}


def export_llm_config(
    plan: families.XnnpackPlan,
    params_path: Path,
    checkpoint: Path,
    output: Path,
    context: int,
    prefill_chunk: int,
    recipe: settings.XnnpackRecipe,
    bos: int | None,
    eos: list[int],
    backend: str = BACKEND,
) -> dict:
    extra = {}
    if bos is not None:
        extra["get_bos_id"] = bos
    if eos:
        extra["get_eos_ids"] = eos
    return {
        "base": {
            "model_class": plan.model_class,
            "params": str(params_path),
            "checkpoint": str(checkpoint),
            "metadata": json.dumps(extra),
        },
        "model": {
            "use_kv_cache": True,
            "use_sdpa_with_kv_cache": True,
            "dtype_override": "fp32",
            "enable_dynamic_shape": True,
        },
        "quantization": {
            "qmode": recipe.qmode,
            "group_size": recipe.group_size,
            "embedding_quantize": recipe.embedding_quantize,
            # Only the embedding reads this. torchao's HQQ search for its scales is
            # platform-dependent, and a hosted runner is whatever GitHub gives the job, so
            # with it on the same model built twice differs in the embedding's low bits and
            # a small model decodes different text (finding 27).
            "use_hqq": recipe.embedding_hqq,
        },
        "export": {
            "max_seq_length": min(prefill_chunk, context),
            "max_context_length": context,
            "output_name": str(output),
        },
        "backend": BACKEND_CONFIG[backend],
    }


def run(
    model_id: str,
    revision: str,
    out_dir: Path,
    work_dir: Path,
    context: int | None = None,
    keep_work: bool = False,
    skip_smoke: bool = False,
    backend: str = BACKEND,
    codes: Path | None = None,
) -> dict:
    if backend not in BACKEND_CONFIG:
        raise ExportError(f"unknown backend {backend!r}; this exporter builds {sorted(BACKEND_CONFIG)}")
    cfg = settings.load()
    source = hub.fetch(model_id, revision)
    verdict = eligibility.evaluate(source, cfg)
    if verdict.reasons:
        raise ExportError(f"{model_id} is not eligible: {'; '.join(verdict.reasons)}")
    if verdict.backends[backend] is not None:
        raise ExportError(f"{model_id} cannot be exported to {backend}: {verdict.backends[backend]}")

    family = families.family_for(source.config)
    plan = families.xnnpack_plan(family, source.config)
    arch = families.architecture(source.config, source.total_params)
    host = host_info()
    budget = host_budget(host)
    choice = sizing.choose_context(arch, cfg.context_tiers, cfg.device_budget_bytes, cfg.runtime_overhead_bytes, budget)
    # Every tier is exported, one job per window; the app and the benchmarker decide what
    # fits a phone (the estimate goes into the report as fits_phone_budget). Only the host
    # limit is a gate: past it the runner VM is killed outright, with no Python error.
    if context is None:
        exportable = [row["context"] for row in choice.table if row["fits_host"]]
        if not exportable:
            raise SkipExport(f"no window in {list(cfg.context_tiers)} can be exported on this host ({budget:,} B)")
        window = exportable[0]
        window_reason = f"largest window this host can export: {window}"
    else:
        window = context
        window_reason = f"requested window {context}"
        peak = sizing.export_peak_bytes(arch, context)
        if budget is not None and peak > budget:
            raise SkipExport(
                f"a {context}-token export needs about {peak:,} B (causal masks alone "
                f"{sizing.causal_mask_bytes(arch, context):,} B); this host has {budget:,} B"
            )
    resident = sizing.device_resident_bytes(arch, window, cfg.runtime_overhead_bytes)

    recipe = cfg.xnnpack if backend == "xnnpack" else cfg.vulkan
    output_repo = naming.output_repo(model_id, cfg.hub_org, cfg.repo_suffix)
    pte_name = naming.cpu_gpu_file(model_id, backend, recipe.qmode, window)
    pte_path_in_repo = f"{backend}/{pte_name}"
    problems = naming.check_app_rules(output_repo, pte_path_in_repo, backend)
    if problems:
        raise ExportError("; ".join(problems))

    out_dir.mkdir(parents=True, exist_ok=True)
    backend_dir = out_dir / backend
    backend_dir.mkdir(exist_ok=True)
    src_dir = work_dir / "source"
    checkpoint = work_dir / "checkpoint" / "consolidated.pth"
    print(f"==> {model_id}@{source.sha[:12]}: family {family.key}, class {plan.model_class}, window {window}")

    started = time.time()
    hub.download(source, src_dir)
    tokenizer, licenses = copy_side_files(source, src_dir, out_dir)

    print("==> converting checkpoint")
    conversion = convert.convert(src_dir, checkpoint, plan.converter, families.text_config(source.config))
    for weights in src_dir.glob("*.safetensors"):
        weights.unlink()  # free the disk before export writes the .pte
    convert_peak = self_peak_rss()

    params_path = work_dir / "params.json"
    params_path.write_text(json.dumps(plan.params, indent=2), encoding="utf-8")
    bos, eos = hub.special_token_ids(source)
    pte = backend_dir / pte_name
    config = export_llm_config(plan, params_path, checkpoint, pte, window, cfg.prefill_chunk, recipe, bos, eos, backend)
    config_path = work_dir / "export_llm.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    # With solved codes, the same export runs through a wrapper that replaces torchao's
    # rounding with them; without, export_llm is called directly, as before.
    if codes is not None:
        command = [
            sys.executable,
            "-m",
            "pipeline.export_with_codes",
            "--config",
            str(config_path),
            "--codes",
            str(codes),
        ]
        print(f"==> export_llm with calibrated codes from {codes}")
    else:
        command = [sys.executable, "-m", "executorch.extension.llm.export.export_llm", "--config", str(config_path)]
        print("==> export_llm")
    export_started = time.time()
    with MemorySampler() as memory:
        run_tool(
            command,
            "export_llm",
            cwd=work_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
        )
    export_seconds = time.time() - export_started
    if not pte.exists():
        raise ExportError(f"export_llm finished but {pte} does not exist")
    if not keep_work:
        checkpoint.unlink(missing_ok=True)

    print("==> reading metadata")
    method_values = smoke.read_metadata(pte)
    mismatches = []
    if method_values.get("get_max_context_len") != window:
        mismatches.append(f"get_max_context_len {method_values.get('get_max_context_len')} != {window}")
    if method_values.get("get_max_seq_len") != min(cfg.prefill_chunk, window):
        mismatches.append(f"get_max_seq_len {method_values.get('get_max_seq_len')}")
    if bos is not None and method_values.get("get_bos_id") != bos:
        mismatches.append(f"get_bos_id {method_values.get('get_bos_id')} != {bos}")
    if eos and method_values.get("get_eos_ids") != eos:
        mismatches.append(f"get_eos_ids {method_values.get('get_eos_ids')} != {eos}")
    if mismatches:
        raise ExportError("exported metadata disagrees with the request: " + "; ".join(mismatches))

    smoke_result = None
    if backend == "vulkan":
        print("==> structural check")
        smoke_result = smoke.structural_check(pte, ("forward",), DELEGATE_ID[backend])
    elif not skip_smoke:
        print("==> smoke test")
        smoke_result = smoke.run(
            pte,
            out_dir / tokenizer,
            src_dir,
            source.tokenizer_config,
            instruct=verdict.variant == "instruct",
            total_params=arch.total_params,
        )
        print(f"    reply: {smoke_result['reply']!r}")

    kv_per_token = sizing.kv_cache_bytes(arch, 1)
    report = {
        "backend": backend,
        "target": None,
        "tokenizer": tokenizer,
        "output_repo": output_repo,
        "source": source_report(source, family.key, verdict.variant, licenses),
        "toolchain": toolchain(),
        "recipe": {
            "model_class": plan.model_class,
            "converter": plan.converter,
            "params": plan.params,
            "qmode": recipe.qmode,
            "group_size": recipe.group_size,
            "embedding_quantize": recipe.embedding_quantize,
            "embedding_hqq": recipe.embedding_hqq,
            "int4_codes": "gptq" if codes is not None else "round-to-nearest",
            "prefill_chunk": min(cfg.prefill_chunk, window),
            "kv_cache_dtype": "fp32",
            "label": f"{recipe.qmode}-g{recipe.group_size}, int8 embeddings",
            "description": (
                f"ExecuTorch {toolchain()['executorch']} `export_llm`: 8-bit dynamic activations "
                f"and 4-bit weights in groups of {recipe.group_size}, int8 per-channel embeddings, "
                f"{'XNNPACK with extended ops' if backend == 'xnnpack' else 'the Vulkan delegate'}, "
                f"prefill chunk {min(cfg.prefill_chunk, window)}, fp32 KV cache."
            ),
        },
        "window": {
            "context": window,
            "reason": window_reason,
            "kv_cache_bytes_per_token": kv_per_token,
            "device_resident_bytes": resident,
            "phone_budget_bytes": cfg.device_budget_bytes,
            "fits_phone_budget": resident <= cfg.device_budget_bytes,
            "table": list(choice.table),
        },
        "files": [
            {"path": pte_path_in_repo, "bytes": pte.stat().st_size, "sha256": sha256(pte)},
        ],
        "estimates": {
            "pte_bytes_estimate": sizing.pte_bytes_estimate(arch, window),
            "pte_bytes_actual": pte.stat().st_size,
            "export_peak_bytes_estimate": sizing.export_peak_bytes(arch, window),
        },
        "host": {
            **host,
            "peak_rss_convert_bytes": convert_peak,
            "peak_rss_export_bytes": children_peak_rss(),
            **memory.result(),
            "export_seconds": round(export_seconds, 1),
            "total_seconds": round(time.time() - started, 1),
        },
        "conversion": conversion,
        "metadata": method_values,
        "smoke": smoke_result,
        "run": manifest.run_info(),
    }
    (backend_dir / "config.json").write_text(
        json.dumps(manifest.backend_config(report), indent=2) + "\n", encoding="utf-8"
    )
    (backend_dir / report_file(window)).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)
    if smoke_result is not None and not smoke_result["passed"]:
        raise ExportError("smoke test failed: " + "; ".join(smoke_result["problems"]))
    return report
