"""Loads config/pipeline.yaml and config/versions.env."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


@dataclass(frozen=True)
class XnnpackRecipe:
    qmode: str
    group_size: int
    embedding_quantize: str
    embedding_hqq: bool


@dataclass(frozen=True)
class QnnRecipe:
    socs: tuple[str, ...]
    model_mode: str
    prefill_ar_len: int
    max_context_len: int
    calib_tasks: tuple[str, ...]
    calib_limit: int


@dataclass(frozen=True)
class RunnerTier:
    """A CI runner size, and what it can carry. Smallest first in MtkRecipe.runner_tiers."""

    label: str
    ram_bytes: int
    disk_bytes: int
    swap_gib: int


@dataclass(frozen=True)
class MtkRecipe:
    socs: tuple[str, ...]
    precision: str
    max_chunks: int
    prompt_tokens: int
    cache_size: int
    calibration: str
    response_cap: int
    min_calibration_prompts: int
    runner_tiers: tuple[RunnerTier, ...] = ()


@dataclass(frozen=True)
class Settings:
    hub_org: str
    repo_suffix: str
    hub_tags: tuple[str, ...]
    watch_orgs: tuple[str, ...]
    org_families: dict[str, tuple[str, ...]]
    limit_per_org: int
    max_dispatch_per_run: int
    pipeline_tag: str
    max_nominal_billions: float
    max_params: int
    name_exclude: tuple[str, ...]
    context_tiers: tuple[int, ...]
    device_budget_bytes: int
    runtime_overhead_bytes: int
    prefill_chunk: int
    xnnpack: XnnpackRecipe
    vulkan: XnnpackRecipe
    qnn: QnnRecipe
    mtk: MtkRecipe
    executorch_version: str


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@lru_cache(maxsize=1)
def load() -> Settings:
    raw = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    versions = read_env_file(CONFIG_DIR / "versions.env")
    hub, watch, export = raw["hub"], raw["watch"], raw["export"]
    tiers = tuple(sorted((int(t) for t in export["context_tiers"]), reverse=True))
    return Settings(
        hub_org=hub["org"],
        repo_suffix=hub["repo_suffix"],
        hub_tags=tuple(hub["tags"]),
        watch_orgs=tuple(watch["org_families"]),
        org_families={org: tuple(tokens) for org, tokens in watch["org_families"].items()},
        limit_per_org=int(watch["limit_per_org"]),
        max_dispatch_per_run=int(watch["max_dispatch_per_run"]),
        pipeline_tag=watch["pipeline_tag"],
        max_nominal_billions=float(watch["max_nominal_billions"]),
        max_params=int(watch["max_params"]),
        name_exclude=tuple(s.lower() for s in watch["name_exclude"]),
        context_tiers=tiers,
        device_budget_bytes=int(export["device_budget_bytes"]),
        runtime_overhead_bytes=int(export["runtime_overhead_bytes"]),
        prefill_chunk=int(export["prefill_chunk"]),
        xnnpack=XnnpackRecipe(
            qmode=export["xnnpack"]["qmode"],
            group_size=int(export["xnnpack"]["group_size"]),
            embedding_quantize=str(export["xnnpack"]["embedding_quantize"]),
            embedding_hqq=bool(export["xnnpack"].get("embedding_hqq", False)),
        ),
        vulkan=XnnpackRecipe(
            qmode=export["vulkan"]["qmode"],
            group_size=int(export["vulkan"]["group_size"]),
            embedding_quantize=str(export["vulkan"]["embedding_quantize"]),
            embedding_hqq=bool(export["vulkan"].get("embedding_hqq", False)),
        ),
        qnn=QnnRecipe(
            socs=tuple(export["qnn"]["socs"]),
            model_mode=export["qnn"]["model_mode"],
            prefill_ar_len=int(export["qnn"]["prefill_ar_len"]),
            max_context_len=int(export["qnn"]["max_context_len"]),
            calib_tasks=tuple(export["qnn"]["calib_tasks"]),
            calib_limit=int(export["qnn"]["calib_limit"]),
        ),
        mtk=MtkRecipe(
            socs=tuple(export["mtk"]["socs"]),
            precision=export["mtk"]["precision"],
            max_chunks=int(export["mtk"]["max_chunks"]),
            prompt_tokens=int(export["mtk"]["prompt_tokens"]),
            cache_size=int(export["mtk"]["cache_size"]),
            calibration=export["mtk"]["calibration"],
            response_cap=int(export["mtk"]["response_cap"]),
            min_calibration_prompts=int(export["mtk"]["min_calibration_prompts"]),
            runner_tiers=tuple(
                RunnerTier(
                    label=str(tier["label"]),
                    ram_bytes=int(tier["ram_bytes"]),
                    disk_bytes=int(tier["disk_bytes"]),
                    swap_gib=int(tier["swap_gib"]),
                )
                for tier in export["mtk"].get("runner_tiers", ())
            ),
        ),
        executorch_version=versions["EXECUTORCH_VERSION"],
    )
