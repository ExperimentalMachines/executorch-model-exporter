"""The files published beside each export: config.json, README.md, NOTICE."""

from __future__ import annotations

import os

from pipeline import naming

# Methods the app reads from config.json before download (openweights ExportConfig.kt),
# plus the rest of the runtime's LLM metadata, as the .pte reports them.
CONFIG_METHODS = (
    "get_max_context_len",
    "get_max_seq_len",
    "get_bos_id",
    "get_eos_ids",
    "get_vocab_size",
    "get_n_layers",
    "use_kv_cache",
    "use_sdpa_with_kv_cache",
    "enable_dynamic_shape",
    # LFM2 only, and the reason it is here: a host reads this to tell an export that clears
    # its short-convolution state from one that leaks it between prompts, and the app reads
    # config.json before it downloads anything (pipeline/lfm2_state.py).
    "get_state_reset_at_zero",
)

# Attribution the upstream licenses require of redistributed derivatives, keyed by the
# app family token of the source name.
NOTICES = {
    # Llama 3.2 Community License, section 1.b.iii.
    "llama32": (
        "Llama 3.2 is licensed under the Llama 3.2 Community License, "
        "Copyright © Meta Platforms, Inc. All Rights Reserved.\n"
    ),
    # Gemma Terms of Use, section 3.1.
    "gemma3": ("Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n"),
}
BUILT_WITH = {"llama32": "Built with Llama"}

BACKEND_TITLES = {
    "xnnpack": "XNNPACK (CPU)",
    "vulkan": "Vulkan (GPU)",
    "qnn": "Qualcomm QNN (HTP)",
    "mtk": "MediaTek NeuroPilot",
}


def _variants(report: dict) -> list[dict]:
    window = report["window"]
    common = {
        "context": window["context"],
        "source_revision": report["source"]["sha"],
        "quantization": report["recipe"]["label"],
        # The sizing estimate against the phone budget (None for the NPU backends, whose
        # runtime memory is not modelled here); the app and the benchmarker decide.
        "fits_phone_budget": window.get("fits_phone_budget"),
    }
    if report["backend"] == "mtk":
        # One model in several files: the chunks run in order, plus the embedding table.
        # MediaTek's LLM runner (examples/mediatek/executor_runner) takes ``runner`` as flags.
        return [
            {
                "files": [f["path"].rsplit("/", 1)[-1] for f in report["files"] if f["path"].endswith(".pte")],
                "embedding": report["runner"]["token_embedding_path"],
                "size_bytes": sum(f["bytes"] for f in report["files"]),
                "sha256": {f["path"].rsplit("/", 1)[-1]: f["sha256"] for f in report["files"]},
                **common,
                "methods": {},
                "runner": report["runner"],
            }
        ]
    variants = []
    for f in report["files"]:
        if not f["path"].endswith(".pte"):
            continue
        methods = {k: report["metadata"][k] for k in CONFIG_METHODS if k in report["metadata"]}
        variants.append(
            {
                "file": f["path"].rsplit("/", 1)[-1],
                "size_bytes": f["bytes"],
                "sha256": f["sha256"],
                **common,
                "methods": methods,
            }
        )
    return variants


def backend_config(reports: dict | list[dict]) -> dict:
    """config.json for one backend folder, in the variants form the app reads: one variant
    per exported window, smallest first, from that folder's reports."""
    reports = [reports] if isinstance(reports, dict) else list(reports)
    reports.sort(key=lambda r: r["window"]["context"])
    report = reports[-1]  # the newest layout wins for the shared fields
    variants = [variant for r in reports for variant in _variants(r)]
    config = {
        "runtime": "executorch",
        "runtime_version": report["toolchain"]["executorch"],
        "backend": report["backend"],
        "target": report.get("target"),
        "tokenizer": report["tokenizer"],
        "source_model": report["source"]["id"],
        "source_revision": report["source"]["sha"],
        "variants": variants,
    }
    if report["toolchain"].get("qairt"):
        # HTP context binaries only load on the QNN runtime they were compiled with.
        config["qnn_sdk_version"] = report["toolchain"]["qairt"]
    if report["backend"] == "mtk":
        config["neuropilot_sdk"] = report["neuropilot"]
    return config


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def _card_metadata(source: dict, tags: list[str]) -> str:
    lic = source.get("license") or {}
    lines = ["---"]
    if lic.get("license"):
        lines.append(f"license: {lic['license']}")
    for key in ("license_name", "license_link"):
        if lic.get(key):
            lines.append(f"{key}: {lic[key]}")
    lines += [
        "base_model:",
        f"- {source['id']}",
        "base_model_relation: quantized",
        "library_name: executorch",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"- {t}" for t in tags],
        "---",
    ]
    return "\n".join(lines)


def readme(repo_id: str, reports: list[dict], hub_tags: list[str], license_files: list[str]) -> str:
    """Model card covering every backend present in the repo."""
    reports = sorted(
        reports,
        key=lambda r: (list(BACKEND_TITLES).index(r["backend"]), r.get("target") or "", r["window"]["context"]),
    )
    source = reports[0]["source"]
    token = naming.app_family(naming.source_name(source["id"]))
    tags = sorted(set(hub_tags) | {r["backend"] for r in reports})
    upstream = f"https://huggingface.co/{source['id']}"
    revisions = sorted({r["source"]["sha"] for r in reports})
    out = [_card_metadata(source, tags), ""]
    if token in BUILT_WITH:
        out += [f"**{BUILT_WITH[token]}**", ""]
    revision_text = (
        f"revision `{revisions[0][:12]}`"
        if len(revisions) == 1
        else "revisions " + ", ".join(f"`{sha[:12]}`" for sha in revisions) + " (see each variant's `source_revision`)"
    )
    out += [
        f"# {naming.source_name(source['id'])} for ExecuTorch",
        "",
        f"ExecuTorch exports of [{source['id']}]({upstream}) ({revision_text}) "
        "for on-device inference with the "
        "[openweights](https://github.com/alpharomercoma/openweights) Android app or any "
        f"ExecuTorch {reports[0]['toolchain']['executorch']} runtime.",
        "",
        "## Files",
        "",
        "Every backend is exported at every context window the runner could build (2k to 32k"
        + (", plus MediaTek's default 512" if any(r["backend"] == "mtk" for r in reports) else "")
        + "). "
        "The window is fixed inside the file: the runtime allocates the whole KV cache at load, "
        "so pick the largest window the device can hold (`fits_phone_budget` in each folder's "
        "`config.json` is the estimate against a 5 GB budget).",
        "",
        "| Backend | Target | File | Window | Size | Smoke test |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        smoke = r.get("smoke") or {}
        verdict = "passed" if smoke.get("passed") else ("not run" if not smoke else "failed")
        if smoke.get("kind") == "structural":
            verdict = "structure checked (no host NPU runtime)" if smoke.get("passed") else "structure check failed"
        if smoke.get("answered"):
            verdict += ' ("Paris")'
        if smoke.get("template_error"):
            verdict += " with a completion prompt (the chat template did not render)"
        for f in r["files"]:
            if f["path"].endswith(".pte"):
                out.append(
                    f"| {BACKEND_TITLES[r['backend']]} | {target(r)} | "
                    f"[`{f['path']}`]({f['path']}) | {r['window']['context']:,} tokens | "
                    f"{_gb(f['bytes'])} | {verdict} |"
                )
    embeddings = sorted({f["path"] for r in reports for f in r["files"] if f["path"].endswith(".bin")})
    if embeddings:
        out += [
            "",
            "MediaTek folders also hold the token embedding table the NeuroPilot runner reads from disk, "
            "shared by every window: " + ", ".join(f"[`{p}`]({p})" for p in embeddings) + ".",
        ]
    out += [
        "",
        f"Tokenizer: [`{reports[0]['tokenizer']}`]({reports[0]['tokenizer']}), copied unchanged "
        "from the source repo. Each backend folder has a `config.json` listing every window as a "
        "variant with the metadata the `.pte` reports, and an `export-report-<window>.json` per "
        "file with the full export record.",
        "",
        "## Memory",
        "",
    ]
    for r in reports:
        kv = r["window"].get("kv_cache_bytes_per_token")
        if kv:
            ctx = r["window"]["context"]
            out.append(
                f"- {BACKEND_TITLES[r['backend']]} at {ctx:,} tokens: the KV cache costs {kv:,} bytes per "
                f"token (fp32), {kv * ctx:,} bytes for the whole window, allocated in full when the "
                "model loads."
            )
    out += ["", "## How it was made", ""]
    described = set()
    for r in reports:
        recipe = r["recipe"]
        # One line per distinct recipe text: the same for every XNNPACK window, but
        # MediaTek's names its window and calibration prompts, which differ per window.
        key = (r["backend"], r.get("target"), recipe["description"])
        if key in described:
            continue
        described.add(key)
        line = f"- {BACKEND_TITLES[r['backend']]} {target(r)}: {recipe['description']}"
        same = [x for x in reports if (x["backend"], x.get("target"), x["recipe"]["description"]) == key]
        runs = sorted({x["run"]["url"] for x in same if x.get("run", {}).get("url")})
        if runs:
            line += " Built by " + ", ".join(f"[run {i + 1}]({url})" for i, url in enumerate(runs)) + "."
        out.append(line)
    lic = source.get("license") or {}
    name = lic.get("license_name") or lic.get("license") or "the upstream license"
    out += [
        "",
        "## License",
        "",
        f"A quantized derivative of [{source['id']}]({upstream}), distributed under the same terms ({name}).",
    ]
    if license_files:
        out.append(
            "The upstream license files are included unchanged: "
            + ", ".join(f"[`{f}`]({f})" for f in license_files)
            + "."
        )
    if token in NOTICES:
        out.append("See [`NOTICE`](NOTICE) for the attribution the license requires.")
    qnn = [r for r in reports if r["backend"] == "qnn"]
    if qnn:
        version = qnn[0]["toolchain"].get("qairt")
        out += [
            "",
            f"The `qnn/` folders hold QNN HTP context binaries compiled with the Qualcomm AI Runtime "
            f"SDK (QAIRT) {version} from Qualcomm Technologies, Inc., used under its AI Stack License. "
            "No Qualcomm SDK or runtime library is included; running them needs the matching QNN "
            "runtime (for example `executorch-android-qnn` 1.4.0, which depends on `qnn-runtime` 2.37.0).",
        ]
    mtk = [r for r in reports if r["backend"] == "mtk"]
    if mtk:
        sdk = mtk[0]["neuropilot"]
        out += [
            "",
            f"The `mtk/` folders hold model binaries compiled with the MediaTek NeuroPilot Express SDK "
            f"{sdk['build']} (mtk_converter {sdk['mtk_converter']}, mtk_neuron {sdk['mtk_neuron']}) "
            "from MediaTek Inc., used under MediaTek's license terms for that SDK. No MediaTek SDK or "
            "runtime library is included. They run on MediaTek's LLM runner from ExecuTorch "
            "(`examples/mediatek/executor_runner`) with the device's NeuroPilot runtime; the settings it "
            "needs are in each folder's `config.json`, under each variant's `runner`.",
        ]
    return "\n".join(out) + "\n"


def target(report: dict) -> str:
    if not report.get("target"):
        return "any arm64"
    name = report.get("target_name")
    return f"{report['target'].upper()} ({name})" if name else report["target"].upper()


def run_info() -> dict:
    server = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if server and repo and run_id:
        return {"url": f"{server}/{repo}/actions/runs/{run_id}", "id": run_id}
    return {}
