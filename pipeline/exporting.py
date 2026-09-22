"""Helpers every backend's export uses: host facts, hashing, side files, report pieces."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from importlib import metadata as pkg_metadata
from pathlib import Path

from pipeline import hub, manifest, naming

TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model")


class ExportError(Exception):
    pass


class SkipExport(ExportError):
    """This window cannot be exported on this host (memory), which is a property of the
    runner, not of the model: the CLI exits 4 and the workflow records a skip, not a failure."""


def report_file(context: int) -> str:
    """One report per window, next to the files it describes: ``export-report-8k.json``."""
    return f"export-report-{naming.window_label(context)}.json"


def run_tool(command: list[str], what: str, **kwargs) -> None:
    """Run an export tool, turning a non-zero exit into ExportError (exit code 2, not a
    traceback) for the CLI. Its own output already says what went wrong."""
    import subprocess

    try:
        subprocess.run(command, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise ExportError(f"{what} exited with status {error.returncode}; its output above says why") from None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contains(path: Path, needle: bytes, block: int = 1 << 24) -> bool:
    """Whether ``needle`` occurs in the file, scanning in blocks (a .pte can be gigabytes)."""
    tail = b""
    with path.open("rb") as f:
        while chunk := f.read(block):
            window = tail + chunk
            if needle in window:
                return True
            tail = window[-(len(needle) - 1) :]
    return False


def host_info() -> dict:
    info = {"nproc": os.cpu_count()}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        fields = {}
        for line in meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = int(rest.split()[0]) * 1024
        info["mem_total_bytes"] = fields.get("MemTotal")
        info["swap_total_bytes"] = fields.get("SwapTotal")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        # Hosted runners differ: the same Qwen3-0.6B calibration took 1,761 s on one and
        # 936 s on another (docs/research/export-bottlenecks.md), so reports name the CPU.
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.partition(":")[2].strip()
                break
    return info


class MemorySampler:
    """Samples /proc/meminfo while an export runs.

    Peak RSS stops at physical memory, so it cannot show how far an export went into swap:
    the Qwen3-0.6B QNN exports at 2k and 4k both peaked at ~15.7 GB RSS on a 16.8 GB runner
    (docs/research/export-bottlenecks.md). This records the swap actually in use, and the
    peak of RAM plus swap in use within one sample, with its time for matching against the
    log: peak swap and lowest MemAvailable alone only bound it, since they need not coincide
    (docs/research, finding 23).
    """

    def __init__(self, interval: float = 5.0, meminfo: Path = Path("/proc/meminfo")):
        self.interval = interval
        self.meminfo = meminfo
        self.peak_swap_used = None
        self.min_available = None
        self.peak_in_use = None
        self.peak_in_use_at = None
        self._stop = threading.Event()
        self._thread = None

    def sample(self) -> None:
        if not self.meminfo.exists():
            return
        fields = {}
        for line in self.meminfo.read_text().splitlines():
            key, _, rest = line.partition(":")
            if rest.split():
                fields[key] = int(rest.split()[0]) * 1024
        swap_used = fields.get("SwapTotal", 0) - fields.get("SwapFree", 0)
        available = fields.get("MemAvailable")
        self.peak_swap_used = max(swap_used, self.peak_swap_used or 0)
        if available is not None:
            self.min_available = available if self.min_available is None else min(available, self.min_available)
            in_use = fields.get("MemTotal", 0) - available + swap_used
            if self.peak_in_use is None or in_use > self.peak_in_use:
                self.peak_in_use = in_use
                self.peak_in_use_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> MemorySampler:
        self.sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.sample()

    def result(self) -> dict:
        return {
            "peak_swap_used_bytes": self.peak_swap_used,
            "min_mem_available_bytes": self.min_available,
            "peak_in_use_bytes": self.peak_in_use,
            "peak_in_use_at": self.peak_in_use_at,
        }


# What the host keeps for itself: the kernel, the runner agent and the shell the export
# runs under. Subtracted from every budget so a plan that "just fits" is not counting it.
RESERVE_BYTES = 1_000_000_000


def host_budget(info: dict, reserve: int = RESERVE_BYTES) -> int | None:
    if info.get("mem_total_bytes") is None:
        return None
    return info["mem_total_bytes"] + (info.get("swap_total_bytes") or 0) - reserve


def children_peak_rss() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    # ru_maxrss is in kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024


def self_peak_rss() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def toolchain(*extra: str) -> dict:
    versions = {}
    for package in ("executorch", "torch", "torchao", *extra):
        try:
            versions[package] = pkg_metadata.version(package)
        except pkg_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def copy_side_files(source: hub.SourceModel, src_dir: Path, out_dir: Path) -> tuple[str, list[str]]:
    """Tokenizer, license files and NOTICE into the repo root; returns (tokenizer, licenses).

    The tokenizer only ever goes to the root: if any backend folder carried its own, the
    app would stop lending the root one to the other folders (HuggingFaceClient.tokenizerFor).
    """
    for name in TOKENIZER_FILES:
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, out_dir / name)
            tokenizer = name
            break
    else:
        raise ExportError("source repo has neither tokenizer.json nor tokenizer.model")
    licenses = []
    for name in source.license_files:
        if (src_dir / name).exists():
            shutil.copy2(src_dir / name, out_dir / name)
            licenses.append(name)
    token = naming.app_family(naming.source_name(source.id))
    if token in manifest.NOTICES:
        (out_dir / "NOTICE").write_text(manifest.NOTICES[token], encoding="utf-8")
    return tokenizer, licenses


def source_report(source: hub.SourceModel, family: str | None, variant: str, licenses: list[str]) -> dict:
    return {
        "id": source.id,
        "sha": source.sha,
        "created_at": source.created_at.isoformat() if source.created_at else None,
        "total_params": source.total_params,
        "family": family,
        "variant": variant,
        "license": source.card_license,
        "license_files": licenses,
    }
