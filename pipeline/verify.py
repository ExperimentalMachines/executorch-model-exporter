"""Smoke-test an already exported .pte, on a host that can run ExecuTorch's runner.

    python -m pipeline verify <out dir> --backend xnnpack

Split out of the export because the two steps do not need the same machine. The export is
arm64-clean: on Ubuntu ARM the download, the conversion and ``export_llm`` all finish. What
does not work there is the runtime, which raises ``IndexError: stoi`` from inside the
wheel's ``TextLLMRunner`` (finding 30), so the export can have the cheaper ARM runner while
the test that makes the file trustworthy runs where the runner works.

It is the same test, not a weaker one. The file is loaded by the same C++ runner the app
uses, asked the same question, and has to answer it; ``passed: false`` fails the job, so a
file that cannot generate is never published. The result is written into the export report
that the export left behind, because that report is the contract publish and summary read.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import families, gate, hub, settings, smoke
from pipeline.exporting import ExportError


def reports(out_dir: Path, backend: str) -> list[Path]:
    found = sorted((out_dir / backend).glob("export-report-*.json"))
    if not found:
        raise ExportError(f"no export report under {out_dir / backend}; nothing to verify")
    return found


def run(out_dir: Path, backend: str, work_dir: Path, gate_reference: Path | None = None) -> dict:
    """Run the smoke test for every window in ``out_dir`` and write the results back."""
    results = []
    gates: list[dict] = []
    for path in reports(out_dir, backend):
        report = json.loads(path.read_text())
        if report.get("smoke") is not None:
            print(f"==> {path.name}: already verified, leaving it")
            results.append(report["smoke"])
            continue

        window = report["window"]["context"]
        # "files" is a list; its one entry's path is relative to the repository root.
        pte = out_dir / report["files"][0]["path"]
        if not pte.exists():
            raise ExportError(f"{path.name} names {pte}, which is not in this directory")

        # The chat template and the parameter count come from the source repo, not the
        # export, so the small files are fetched again here. The weights are not: only the
        # tokenizer and the configs, which is seconds rather than gigabytes.
        source = hub.fetch(report["source"]["id"], report["source"]["sha"])
        src_dir = work_dir / "source"
        hub.download(source, src_dir, weights=False)
        arch = families.architecture(source.config, source.total_params)

        print(f"==> smoke test, window {window}")
        result = smoke.run(
            pte,
            out_dir / report["tokenizer"],
            src_dir,
            source.tokenizer_config,
            instruct=report["source"].get("variant") == "instruct",
            total_params=arch.total_params,
        )
        report["smoke"] = result
        report["measuring_gate"] = gate.measuring_gate(result.get("stats"))
        if report["measuring_gate"]["prefill_tok_per_sec"]:
            print(
                f"    measuring gate: prefill {report['measuring_gate']['prefill_tok_per_sec']} tok/s, "
                f"decode {report['measuring_gate']['decode_tok_per_sec']} tok/s"
            )

        # Generating is not the same as deciding. A file can answer the smoke question and
        # still have lost the tool call that made it worth exporting, which is what happened
        # on 2026-09-18 and is why this is here rather than left to a person to notice.
        if result.get("passed") and gate_reference is not None:
            from transformers import AutoTokenizer

            reference = json.loads(gate_reference.read_text())
            tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
            print("==> decision gate")
            verdict = gate.check(reference, gate.measure(pte, out_dir / report["tokenizer"], tokenizer, reference))
            print(
                f"    fp32 {verdict['fp32_mean']} against the export's {verdict['export_mean']} "
                f"on {verdict['graded']} rows, same choice on {verdict['agreed']}"
            )
            if not verdict.get("measures_tool_calling", True):
                print(
                    "    note: this model answers the tool-shaped rows the same way it answers "
                    "the quiet ones, so the gate checked that the export tracks its fp32 model "
                    "and not that it still calls tools"
                )
            report["gate"] = verdict
            gates.append(verdict)

        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        results.append(result)

    failed = [r for r in results if not r.get("passed")]
    if failed:
        problems = "; ".join(p for r in failed for p in r.get("problems", []))
        raise ExportError(f"smoke test failed: {problems}")
    regressed = [g for g in gates if not g.get("passed")]
    if regressed:
        problems = "; ".join(p for g in regressed for p in g.get("problems", []))
        raise ExportError(f"decision gate failed: {problems}")
    return {"windows": len(results), "passed": True, "gated": len(gates)}


def main(out_dir: Path, backend: str, work_dir: Path) -> int:
    summary = run(out_dir, backend, work_dir)
    print(json.dumps(summary, indent=2))
    return 0


__all__ = ["main", "run", "settings"]
