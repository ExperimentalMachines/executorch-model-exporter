"""Publishing an export: Hugging Face Hub, GitHub Releases, and the job summary."""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

from pipeline import exporting, hub, manifest, naming, settings

# GitHub rejects release assets of 2 GiB or more.
RELEASE_ASSET_LIMIT = 2 * 1024**3


def _folder(backend: str, target: str | None) -> str:
    return backend if target is None else f"{backend}/{target.lower()}"


def load_report(out_dir: Path, backend: str, target: str | None = None, context: int | None = None) -> dict:
    """The export report in the local backend folder: one run exports one window, so there
    is one unless several windows were exported into the same folder by hand, in which
    case ``context`` picks."""
    folder = out_dir / _folder(backend, target)
    found = sorted(folder.glob("export-report*.json"))
    if context is not None:
        found = [p for p in found if p.name == exporting.report_file(context)]
    if len(found) != 1:
        raise ValueError(f"expected one export report in {folder}, found {[p.name for p in found]}; pass --context")
    return json.loads(found[0].read_text(encoding="utf-8"))


def is_report(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.startswith("export-report") and name.endswith(".json")


# The window label sits right before the extension (or before "-chunk<i>of<n>" for MediaTek),
# so a model whose own name carries a window-like token ("SmolLM2-1.7B-Instruct-16k") is not
# read as that window.
_WINDOW_FILE = re.compile(r"-(?P<label>\d+k|\d+)(?:\.pte|-chunk\d+of\d+\.pte)$")


def window_of(name: str) -> str | None:
    """The window label a published file name carries, or None (config, embedding table)."""
    if name.startswith("export-report-") and name.endswith(".json"):
        return name[len("export-report-") : -len(".json")]
    match = _WINDOW_FILE.search(name)
    return match.group("label") if match else None


def superseded(path: str, folder: str, label: str) -> bool:
    """Whether a file already in the repo is replaced by this run: the same window in the
    same backend folder, or the folder's pre-matrix files that carried no window in their
    name. Other windows' files stay: the folder holds every window side by side."""
    if not path.startswith(f"{folder}/") or "/" in path[len(folder) + 1 :]:
        return False
    name = path[len(folder) + 1 :]
    if name in ("config.json", "export-report.json"):
        return True
    return window_of(name) == label


def publish_hf(
    out_dir: Path, backend: str, target: str | None = None, context: int | None = None, attempts: int = 6
) -> str:
    """Commit this run's window into its backend folder beside the other windows, plus the
    shared root files; regenerate the folder's config.json and the repo README.md from
    every report in the repo.

    Runs publish on their own, possibly at the same moment as another (other windows, other
    backends), so the commit is pinned to the revision the README was rendered from and
    retried on conflict.
    """
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    cfg = settings.load()
    report = load_report(out_dir, backend, target, context)
    # The workflow runs the smoke test in its own job and fails before reaching this, but
    # publish is also a command a person can run by hand, against a directory exported with
    # --skip-smoke. Refusing here is what makes "published" mean "ran and answered" rather
    # than "the workflow happened to be wired correctly that day".
    smoke = report.get("smoke")
    if smoke is None:
        raise ValueError(
            f"{report['files'][0]['path']} has no smoke result: run `python -m pipeline verify` "
            "on this directory before publishing it"
        )
    if not smoke.get("passed"):
        raise ValueError(f"{report['files'][0]['path']} failed its smoke test: {'; '.join(smoke.get('problems', []))}")
    repo_id = report["output_repo"]
    folder = _folder(backend, target)
    label = naming.window_label(report["window"]["context"])
    local_folder = out_dir / folder
    # This window's files, plus the folder's shared ones (config.json, MediaTek's embedding).
    new_paths = {
        f"{folder}/{p.name}": p for p in local_folder.iterdir() if p.is_file() and window_of(p.name) in (None, label)
    }
    root_files = {p.name: p for p in out_dir.iterdir() if p.is_file()}
    scoped = [path for path in new_paths if path.rsplit("/", 1)[-1] in exporting.TOKENIZER_FILES]
    if scoped:
        # A tokenizer inside any backend folder makes the app stop lending the root one to
        # every other folder (HuggingFaceClient.tokenizerFor), breaking the other backends.
        raise ValueError(f"tokenizer files must stay at the repo root, found {scoped}")

    hf = hub.api()
    hf.create_repo(repo_id, repo_type="model", exist_ok=True)

    for attempt in range(attempts):
        info = hf.model_info(repo_id, expand=["sha", "siblings"])
        existing = [s.rfilename for s in info.siblings or []]
        stale = [path for path in existing if superseded(path, folder, label)]
        reports = [report]
        migrated: dict[str, bytes] = {}
        for path in existing:
            if not is_report(path) or (path in stale and not path.endswith("/export-report.json")):
                continue
            local = hf_hub_download(repo_id, path, revision=info.sha, token=hf.token)
            other = json.loads(Path(local).read_text(encoding="utf-8"))
            if path in stale:
                # A pre-matrix report carries no window in its name. Its window's file stays
                # unless it is the one this run replaces, so the report moves to the labelled name.
                if naming.window_label(other["window"]["context"]) == label:
                    continue
                migrated[f"{folder}/{exporting.report_file(other['window']['context'])}"] = (
                    json.dumps(other, indent=2).encode() + b"\n"
                )
            reports.append(other)
        folder_reports = [r for r in reports if _folder(r["backend"], r.get("target")) == folder]
        config = json.dumps(manifest.backend_config(folder_reports), indent=2) + "\n"
        license_files = sorted({f for r in reports for f in r["source"].get("license_files", [])})
        readme = manifest.readme(repo_id, reports, list(cfg.hub_tags), license_files)

        operations = [CommitOperationDelete(path_in_repo=path) for path in stale if path not in new_paths]
        operations += [
            CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v))
            for k, v in new_paths.items()
            if k != f"{folder}/config.json"
        ]
        operations.append(CommitOperationAdd(path_in_repo=f"{folder}/config.json", path_or_fileobj=config.encode()))
        operations += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in migrated.items()]
        operations += [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(v)) for k, v in root_files.items()]
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
        try:
            commit = hf.create_commit(
                repo_id,
                operations,
                commit_message=f"{folder} {label}: {report['source']['id']}@{report['source']['sha'][:12]}",
                parent_commit=info.sha,
            )
            return commit.commit_url
        except HfHubHTTPError as error:
            status = getattr(error.response, "status_code", None)
            # 412 is the parent_commit precondition failing; 409 is the Hub refusing a
            # concurrent commit to the same branch. Both mean another window's job got
            # there first, and both are fixed by rebuilding the commit against the head
            # this loop re-reads at the top. Five windows publish at once, so this is the
            # normal case, not an exceptional one: run 35373841386 lost the 8192 window to
            # an unretried 409 while the other four went through.
            if status not in (409, 412) or attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def release_tag(report: dict) -> str:
    """One release per backend, chip, window and source revision: the window jobs of one
    model finish at the same time and must not race on one tag."""
    name = report["source"]["id"].rsplit("/", 1)[-1]
    target = f"-{report['target']}" if report.get("target") else ""
    label = naming.window_label(report["window"]["context"])
    return f"{name}-{report['backend']}{target}-{label}-{report['source']['sha'][:7]}"


def release_notes(report: dict, hf_url: str | None) -> str:
    lines = [
        f"{manifest.BACKEND_TITLES[report['backend']]} export of "
        f"[{report['source']['id']}](https://huggingface.co/{report['source']['id']}) "
        f"at `{report['source']['sha'][:12]}`, window {report['window']['context']:,} tokens, "
        f"ExecuTorch {report['toolchain']['executorch']}.",
        "",
    ]
    if hf_url:
        lines += [f"Hugging Face: {hf_url}", ""]
    too_big = [f for f in report["files"] if f["bytes"] >= RELEASE_ASSET_LIMIT]
    for f in too_big:
        lines.append(
            f"- `{f['path']}` is {f['bytes']:,} bytes, over GitHub's 2 GiB asset limit: download it from Hugging Face."
        )
    return "\n".join(lines) + "\n"


def publish_release(
    out_dir: Path, backend: str, target: str | None = None, context: int | None = None, attempts: int = 5
) -> str:
    """Create (or update) a GitHub release with this window's files under the 2 GiB limit."""
    report = load_report(out_dir, backend, target, context)
    folder = out_dir / _folder(backend, target)
    tag = release_tag(report)
    label = naming.window_label(report["window"]["context"])
    hf_url = f"https://huggingface.co/{report['output_repo']}"
    candidates = [p for p in sorted(folder.iterdir()) if window_of(p.name) in (None, label)]
    candidates.append(out_dir / report["tokenizer"])
    assets = [str(p) for p in candidates if p.is_file() and p.stat().st_size < RELEASE_ASSET_LIMIT]
    notes = release_notes(report, hf_url)
    create = ["gh", "release", "create", tag, "--title", tag, "--notes", notes, *assets]
    # GitHub's API answers 5xx now and then (a whole evening of them on 2026-09-13); a
    # release is the last step of a multi-hour job, so it is retried, not failed.
    for attempt in range(attempts):
        created = subprocess.run(create, capture_output=True, text=True)
        if created.returncode == 0:
            return tag
        output = created.stderr + created.stdout
        if "already exists" in output:
            break
        if not re.search(r"HTTP 5\d\d", output) or attempt == attempts - 1:
            raise RuntimeError(f"gh release create failed: {created.stderr.strip()[-2000:]}")
        time.sleep(30 * (attempt + 1))
    # The release exists (a re-run of the same window, or a create that answered 5xx after
    # creating it): replace its assets and notes.
    subprocess.run(["gh", "release", "upload", tag, "--clobber", *assets], check=True)
    subprocess.run(["gh", "release", "edit", tag, "--notes", notes], check=True)
    return tag


def summary(out_dir: Path, backend: str, target: str | None = None, context: int | None = None) -> str:
    """Markdown for $GITHUB_STEP_SUMMARY."""
    report = load_report(out_dir, backend, target, context)
    smoke = report.get("smoke") or {}
    host = report["host"]
    estimates = report.get("estimates") or {}
    rows = [
        ("Source", f"{report['source']['id']} @ `{report['source']['sha'][:12]}`"),
        ("Family / variant", f"{report['source']['family']} / {report['source']['variant']}"),
        ("Parameters", f"{report['source']['total_params']:,}"),
        ("Window", f"{report['window']['context']:,} ({report['window']['reason']})"),
        ("Files", ", ".join(f"`{f['path']}` {f['bytes']:,} B" for f in report["files"])),
        ("Peak RSS export", f"{host.get('peak_rss_export_bytes') or 0:,} B"),
        ("Peak swap in use", f"{host.get('peak_swap_used_bytes') or 0:,} B"),
        ("Peak RAM + swap in use", f"{host.get('peak_in_use_bytes') or 0:,} B at {host.get('peak_in_use_at') or '?'}"),
        ("CPU", host.get("cpu_model") or "?"),
        ("Export time", f"{host.get('export_seconds')} s of {host.get('total_seconds')} s"),
    ]
    if estimates:
        rows += [
            (".pte estimate vs actual", f"{estimates['pte_bytes_estimate']:,} / {estimates['pte_bytes_actual']:,} B"),
            ("Export peak estimate", f"{estimates['export_peak_bytes_estimate']:,} B"),
        ]
    if not smoke:
        rows.append(("Check", "not run"))
    elif smoke.get("kind") == "structural":
        rows.append(
            ("Check", "structure " + ("passed" if smoke["passed"] else "failed: " + "; ".join(smoke["problems"])))
        )
    else:
        rows.append(("Smoke test", f"{'passed' if smoke.get('passed') else 'failed'}: {smoke.get('reply', '')!r}"))
        if smoke.get("template_error"):
            rows.append(("Chat template", f"could not be rendered, completion prompt used: {smoke['template_error']}"))
    title = manifest.BACKEND_TITLES[backend] + (f" {manifest.target(report)}" if report.get("target") else "")
    lines = [f"### {title}: {report['output_repo']}", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    return "\n".join(lines) + "\n"
