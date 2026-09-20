"""Gate a file that is already published, without rebuilding it.

    python -m pipeline audit Qwen/Qwen3-1.7B --window 2048

Twelve models were published before the decision gate existed, so nothing has ever checked
that they still decide like the models they were compiled from -- only that they generate,
which is what the smoke test proves and what a file with quietly rounded weights also
passes. Re-exporting them to find out costs a GPTQ solve each, hours per model, to answer a
question about files that already exist.

This asks the question directly: fetch the published .pte, work out what its fp32 source
decides on the gate's prompts, and compare. It publishes nothing and writes nothing to the
Hub. A failure here is a finding about a file that is already out there, which is worth
knowing whether or not anyone re-exports it.

The window matters less than it looks: the gate reads the same on every window of a model
(0.701 on the 2048, 16384 and 32768 exports of LFM2.5-1.2B alike), because the prompts are
far shorter than the smallest window and the weights do not change with it.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import families, gate, hub, naming, settings
from pipeline.exporting import ExportError


def published_repo(model_id: str, cfg) -> str:
    return naming.output_repo(model_id, cfg.hub_org, cfg.repo_suffix)


def _pte_for(hf, repo_id: str, backend: str, window: int | None) -> str:
    files = [f for f in hf.list_repo_files(repo_id) if f.startswith(f"{backend}/") and f.endswith(".pte")]
    if not files:
        raise ExportError(f"{repo_id} publishes no {backend} .pte")
    if window is None:
        return sorted(files)[0]
    label = naming.window_label(window)
    matching = [f for f in files if f.endswith(f"-{label}.pte")]
    if not matching:
        raise ExportError(f"{repo_id} has no {label} file; it publishes {sorted(f.split('/')[-1] for f in files)}")
    return matching[0]


def run(model_id: str, work_dir: Path, backend: str = "xnnpack", window: int | None = None, revision: str = "main"):
    """Gate one published file against the fp32 model it was compiled from."""
    import gc

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = settings.load()
    repo_id = published_repo(model_id, cfg)
    hf = hub.api()
    remote = _pte_for(hf, repo_id, backend, window)

    source = hub.fetch(model_id, revision)
    src_dir = work_dir / "source"
    print(f"==> {model_id} -> {repo_id}/{remote}")
    hub.download(source, src_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
    fp32 = AutoModelForCausalLM.from_pretrained(str(src_dir), dtype=torch.float32).eval()
    print("==> what the unquantised model decides")
    reference = gate.reference(fp32, tokenizer)
    del fp32
    gc.collect()

    print("==> fetching the published file")
    local = Path(hf_hub_download(repo_id, remote, local_dir=str(work_dir / "published")))
    tok_path = Path(hf_hub_download(repo_id, "tokenizer.json", local_dir=str(work_dir / "published")))

    print("==> gate")
    verdict = gate.check(reference, gate.measure(local, tok_path, tokenizer, reference))
    arch = families.architecture(source.config, source.total_params) if source.config else None
    return {
        "model_id": model_id,
        "published_repo": repo_id,
        "file": remote,
        "revision": source.sha,
        "params": arch.total_params if arch else None,
        "gate": verdict,
    }


def main(model_id: str, work_dir: Path, backend: str = "xnnpack", window: int | None = None) -> int:
    report = run(model_id, work_dir, backend, window)
    print(json.dumps(report, indent=2))
    return 0 if report["gate"]["passed"] else 2


__all__ = ["main", "run", "published_repo"]
