"""Command line: ``python -m pipeline <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _plan(args) -> int:
    from pipeline import eligibility, families, hub, settings, sizing
    from pipeline.exporting import host_budget, host_info

    cfg = settings.load()
    source = hub.fetch(args.model, args.revision)
    verdict = eligibility.evaluate(source, cfg)
    result = verdict.to_dict()
    family = families.family_for(source.config) if source.config else None
    if family and source.total_params:
        arch = families.architecture(source.config, source.total_params)
        budget = host_budget(host_info())
        choice = sizing.choose_context(
            arch, cfg.context_tiers, cfg.device_budget_bytes, cfg.runtime_overhead_bytes, budget
        )
        result["windows"] = {
            "exportable_on_this_host": [row["context"] for row in choice.table if row["fits_host"]],
            "within_phone_budget": [row["context"] for row in choice.table if row["fits_device"]],
            "host_budget_bytes": budget,
            "table": list(choice.table),
        }
    print(json.dumps(result, indent=2, default=str))
    return 0 if verdict.eligible else 3


def _run_export(export) -> int:
    """Exit codes: 0 exported, 2 failed, 4 skipped (this host cannot export this window)."""
    from pipeline.exporting import ExportError, SkipExport

    try:
        report = export()
    except SkipExport as error:
        print(f"export skipped: {error}", file=sys.stderr)
        return 4
    except ExportError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"file": report["files"], "window": report["window"]["context"]}, indent=2))
    return 0


def _solve(args) -> int:
    from pipeline import solve as solve_module

    report = solve_module.run(args.model, args.revision, Path(args.out), Path(args.work), damp=args.damp)
    print(json.dumps(report, indent=2))
    return 0


def _export_xnnpack(args) -> int:
    from pipeline import export_xnnpack

    return _run_export(
        lambda: export_xnnpack.run(
            args.model,
            args.revision,
            Path(args.out),
            Path(args.work),
            context=args.context,
            keep_work=args.keep_work,
            skip_smoke=args.skip_smoke,
            codes=Path(args.codes) if args.codes else None,
        )
    )


def _export_vulkan(args) -> int:
    from pipeline import export_xnnpack

    return _run_export(
        lambda: export_xnnpack.run(
            args.model,
            args.revision,
            Path(args.out),
            Path(args.work),
            context=args.context,
            keep_work=args.keep_work,
            backend="vulkan",
        )
    )


def _export_mtk(args) -> int:
    from pipeline import export_mtk

    if not args.tool_python or not args.examples:
        print("export failed: set --tool-python and --examples (or MTK_PYTHON and MTK_EXAMPLES)", file=sys.stderr)
        return 2
    return _run_export(
        lambda: export_mtk.run(
            args.model,
            args.revision,
            args.soc,
            Path(args.out),
            Path(args.work),
            tool_python=args.tool_python,
            examples_dir=Path(args.examples),
            keep_work=args.keep_work,
            context=args.context,
        )
    )


def _export_qnn(args) -> int:
    from pipeline import export_qnn

    return _run_export(
        lambda: export_qnn.run(
            args.model,
            args.revision,
            args.soc,
            Path(args.out),
            Path(args.work),
            keep_work=args.keep_work,
            context=args.context,
        )
    )


def _publish_hf(args) -> int:
    from pipeline import publish

    print(publish.publish_hf(Path(args.out), args.backend, args.target, args.context))
    return 0


def _publish_release(args) -> int:
    from pipeline import publish

    print(publish.publish_release(Path(args.out), args.backend, args.target, args.context))
    return 0


def _summary(args) -> int:
    from pipeline import publish

    sys.stdout.write(publish.summary(Path(args.out), args.backend, args.target, args.context))
    return 0


def _watch(args) -> int:
    from pipeline import settings, watch

    cfg = settings.load()
    state_path = Path(args.state)
    backfill = tuple(m.strip() for m in args.backfill.split(",") if m.strip())
    summary = watch.run(
        state_path,
        cfg,
        dispatch=not args.dry_run,
        backfill=backfill,
        requeue=args.requeue,
        limit_per_org=cfg.limit_per_org,
        max_dispatch=cfg.max_dispatch_per_run,
    )
    if not args.dry_run:
        watch.save_state(state_path, summary["state"])
    text = watch.markdown(summary)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(text)
    print(text)
    return 1 if summary["failed"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="eligibility and window choice for one model, no download")
    plan.add_argument("model")
    plan.add_argument("--revision", default="main")
    plan.set_defaults(func=_plan)

    export = commands.add_parser("export-xnnpack", help="download, convert, export and smoke-test")
    export.add_argument("model")
    export.add_argument("--revision", default="main")
    export.add_argument("--out", default="out")
    export.add_argument("--work", default="work")
    export.add_argument(
        "--context", type=int, default=None, help="window in tokens (default: the largest this host can build)"
    )
    export.add_argument(
        "--keep-work", action="store_true", help="keep the work dir after success (a failure always leaves it)"
    )
    export.add_argument("--skip-smoke", action="store_true")
    export.add_argument(
        "--codes",
        default=None,
        help="codes.pt from `solve`; without it the int4 weights are rounded to nearest",
    )
    export.set_defaults(func=_export_xnnpack)

    solve = commands.add_parser("solve", help="GPTQ the int4 codes once per model, for every window to reuse")
    solve.add_argument("model")
    solve.add_argument("--revision", default="main")
    solve.add_argument("--out", default="codes.pt")
    solve.add_argument("--work", default="work")
    solve.add_argument("--damp", type=float, default=0.01, help="Hessian damping, as a fraction of its mean diagonal")
    solve.set_defaults(func=_solve)

    vulkan = commands.add_parser("export-vulkan", help="download, convert, export a Vulkan (GPU) .pte")
    vulkan.add_argument("model")
    vulkan.add_argument("--revision", default="main")
    vulkan.add_argument("--out", default="out")
    vulkan.add_argument("--work", default="work")
    vulkan.add_argument(
        "--context", type=int, default=None, help="window in tokens (default: the largest this host can build)"
    )
    vulkan.add_argument(
        "--keep-work", action="store_true", help="keep the work dir after success (a failure always leaves it)"
    )
    vulkan.set_defaults(func=_export_vulkan)

    qnn = commands.add_parser("export-qnn", help="compile a Qualcomm HTP .pte for one chip")
    qnn.add_argument("model")
    qnn.add_argument("--soc", required=True, help="e.g. SM8650")
    qnn.add_argument("--revision", default="main")
    qnn.add_argument("--out", default="out")
    qnn.add_argument("--work", default="work")
    qnn.add_argument("--context", type=int, default=None, help="window instead of qnn.max_context_len")
    qnn.add_argument(
        "--keep-work", action="store_true", help="keep the work dir after success (a failure always leaves it)"
    )
    qnn.set_defaults(func=_export_qnn)

    mtk = commands.add_parser("export-mtk", help="compile MediaTek NeuroPilot .pte chunks for one chip")
    mtk.add_argument("model")
    mtk.add_argument("--soc", required=True, help="MT6989 or MT6991")
    mtk.add_argument("--revision", default="main")
    mtk.add_argument("--out", default="out")
    mtk.add_argument("--work", default="work")
    mtk.add_argument("--context", type=int, default=None, help="window instead of mtk.cache_size")
    mtk.add_argument(
        "--tool-python",
        default=os.environ.get("MTK_PYTHON"),
        help="Python 3.10 with requirements/mtk-tools.txt and MediaTek's wheels (env MTK_PYTHON)",
    )
    mtk.add_argument(
        "--examples",
        default=os.environ.get("MTK_EXAMPLES"),
        help="ExecuTorch's examples/mediatek at EXECUTORCH_COMMIT (env MTK_EXAMPLES)",
    )
    mtk.add_argument("--keep-work", action="store_true")
    mtk.set_defaults(func=_export_mtk)

    watch = commands.add_parser("watch", help="check the watched orgs, dispatch exports, update state")
    watch.add_argument("--state", required=True, help="state JSON (on the state branch)")
    watch.add_argument("--backfill", default="", help="comma-separated model ids to evaluate and export now")
    watch.add_argument("--dry-run", action="store_true", help="dispatch nothing and leave the state file alone")
    watch.add_argument("--summary", default=None, help="append the markdown summary to this file")
    watch.add_argument(
        "--requeue", action="store_true", help="put every dispatched export back in the queue (after cancelling runs)"
    )
    watch.set_defaults(func=_watch)

    for name, func, text in (
        ("publish-hf", _publish_hf, "commit one backend folder to the output HF repo"),
        ("publish-release", _publish_release, "attach one backend's files to a GitHub release"),
        ("summary", _summary, "markdown summary of one backend's export"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("out")
        command.add_argument("--backend", required=True, choices=["xnnpack", "vulkan", "qnn", "mtk"])
        command.add_argument("--target", default=None)
        command.add_argument("--context", type=int, default=None, help="which window, when the folder holds several")
        command.set_defaults(func=func)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
