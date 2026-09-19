"""Output names, and the openweights app's name-based rules they have to satisfy.

The app reads a compiled model's family (chat template) and backend from its name, not
from the file, so a wrong name makes a good export unusable. The ``app_*`` functions are
ports of the app's Kotlin so tests can check every name this pipeline generates:

- ``app_family``     PromptTemplates.forModel   (core/.../PromptTemplate.kt)
- ``app_backend``    CompiledBackend.of         (core/.../CompiledBackend.kt)
- ``app_model_name`` ExecuTorchFileName.modelNameFor
- ``app_size_hint``  HubModel.parameterHint     (core/hub/.../HuggingFaceClient.kt)
"""

from __future__ import annotations

import re

APP_FAMILY_TOKENS = (
    # Order matters, exactly as it does in the app: "qwen3" is a prefix of "qwen35", which
    # is a different family with a different template, so it has to be tried first
    # (PromptTemplates.forModel).
    "qwen35",
    "qwen3",
    "qwen25",
    "smollm2",
    "smollm3",
    "llama32",
    "phi4mini",
    "gemma3",
    "lfm25",
)
# Mirrors PromptTemplates.EXCLUDED in the app. "qwen35" was here while the app had no
# template for it; the app has had Qwen35Template since, and keeping the refusal meant this
# repository skipped a family the app can run.
APP_EXCLUDED = ("vl", "vision", "coder", "guard")
_KEPT_PUNCTUATION = "-_."
_SIZE_HINT = re.compile(r"(?<![A-Za-z0-9.])\d+(\.\d+)?[BM](?![A-Za-z0-9])", re.IGNORECASE)

BACKEND_FOLDERS = {"xnnpack": "xnnpack", "vulkan": "vulkan", "qnn": "qnn", "mtk": "mtk"}


def normalise(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def app_family(name: str) -> str | None:
    n = normalise(name)
    if "lfm25vl" in n:
        return "lfm25"
    if any(token in n for token in APP_EXCLUDED):
        return None
    for token in APP_FAMILY_TOKENS:
        if token in n:
            return token
    return None


def app_backend(text: str) -> str:
    name = text.lower()
    if "xnnpack" in name:
        return "xnnpack"
    if "vulkan" in name:
        return "vulkan"
    if "qnn" in name or "qualcomm" in name or "htp" in name:
        return "qnn"
    if "neuropilot" in name or "mediatek" in name or "mtk" in name:
        return "neuropilot"
    if "mlx" in name:
        return "mlx"
    return "unknown"


def _sanitized(text: str) -> str:
    kept = "".join(c if c.isalnum() or c in _KEPT_PUNCTUATION else "-" for c in text)
    return kept.strip("-.") or "model"


def app_model_name(repo_id: str, weights_path: str) -> str:
    repo = _sanitized(repo_id.rsplit("/", 1)[-1])
    file = weights_path.rsplit("/", 1)[-1]
    stem = file[: -len(".pte")] if file.lower().endswith(".pte") else file
    directory = weights_path.rsplit("/", 1)[0] if "/" in weights_path else ""
    directory = _sanitized(directory) if directory else None
    stem = _sanitized(stem) if stem else ""
    distinct = stem if stem and stem.lower() != "model" else directory
    return f"{repo}.pte" if distinct is None else f"{repo}-{distinct}.pte"


def app_size_hint(name: str) -> str | None:
    match = _SIZE_HINT.search(name)
    return match.group(0).upper() if match else None


def size_hints(name: str) -> list[str]:
    """Every size-looking token in the name. The app reads the first; a name with several
    ("Qwen2.5-1M-1.5B") would be shown at the wrong size, so such names are refused."""
    return sorted({m.group(0).upper() for m in _SIZE_HINT.finditer(name)})


def nominal_billions(name: str) -> float | None:
    """The size written in the name, in billions: "1.7B" → 1.7, "360M" → 0.36."""
    hint = app_size_hint(name)
    if hint is None:
        return None
    number = float(hint[:-1])
    return number / 1000 if hint.endswith("M") else number


def source_name(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def output_repo(model_id: str, hub_org: str, suffix: str) -> str:
    return f"{hub_org}/{source_name(model_id)}{suffix}"


def window_label(context: int) -> str:
    if context % 1024 == 0:
        return f"{context // 1024}k"
    return str(context)


def xnnpack_file(model_id: str, qmode: str, context: int) -> str:
    return f"{source_name(model_id)}-{qmode}-{window_label(context)}.pte"


def cpu_gpu_file(model_id: str, backend: str, qmode: str, context: int) -> str:
    """XNNPACK keeps its original name; a Vulkan file says so, since the app reads the
    backend from the name (CompiledBackend.of: "vulkan")."""
    if backend == "xnnpack":
        return xnnpack_file(model_id, qmode, context)
    return f"{source_name(model_id)}-{backend}-{qmode}-{window_label(context)}.pte"


def qnn_folder(soc: str) -> str:
    return f"qnn/{soc.lower()}"


def qnn_file(model_id: str, model_mode: str, context: int) -> str:
    return f"{source_name(model_id)}-qnn-{model_mode}-{window_label(context)}.pte"


def mtk_folder(soc: str) -> str:
    return f"mtk/{soc.lower()}"


def mtk_chunk_file(model_id: str, precision: str, context: int, chunk: int, chunks: int) -> str:
    """One chunk of a MediaTek export; "neuropilot" is what the app reads as the backend."""
    return (
        f"{source_name(model_id)}-neuropilot-{precision.lower()}-{window_label(context)}-chunk{chunk + 1}of{chunks}.pte"
    )


def mtk_embedding_file(model_id: str) -> str:
    """The token embedding table MediaTek's runner reads from disk (fp32, not in the chunks)."""
    return f"{source_name(model_id)}-neuropilot-embedding-fp32.bin"


def check_app_rules(repo_id: str, weights_path: str, backend: str) -> list[str]:
    """Problems that would stop the app from using this file, empty when there are none."""
    problems: list[str] = []
    repo_name = repo_id.rsplit("/", 1)[-1]
    if "xnnpack" in repo_name.lower():
        problems.append(f"repo name {repo_name!r} contains 'xnnpack': the app would read every file in it as XNNPACK")
    detected = app_backend(f"{repo_id}/{weights_path}")
    expected = {"xnnpack": "xnnpack", "vulkan": "vulkan", "qnn": "qnn", "mtk": "neuropilot"}[backend]
    if detected != expected:
        problems.append(f"{repo_id}/{weights_path} reads as backend {detected!r}, not {expected!r}")
    installed = app_model_name(repo_id, weights_path)
    if app_family(installed) is None:
        problems.append(f"installed name {installed!r} matches no chat template the app knows")
    if app_size_hint(repo_name) is None:
        problems.append(f"repo name {repo_name!r} carries no size hint like 1.7B")
    return problems
