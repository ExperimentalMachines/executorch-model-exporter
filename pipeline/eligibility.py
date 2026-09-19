"""Whether a source model should be exported, and to which backends."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pipeline import families, naming
from pipeline.hub import SourceModel
from pipeline.settings import Settings

# Which app template token each family's names must carry.
FAMILY_APP_TOKENS = {
    "qwen3": {"qwen3"},
    "qwen2_5": {"qwen25"},
    "llama": {"llama32", "smollm2"},
    "gemma3": {"gemma3"},
    "smollm3": {"smollm3"},
    # The app has a template for LFM2.5 only. A plain LFM2 release shares the architecture
    # class but reads as no app family at all, so it is refused below rather than exported
    # into a file the app cannot render.
    "lfm2": {"lfm25"},
}
# What a release of each family is called. The architecture class alone is not enough: a
# later generation can keep the class (a text-only "Qwen3.8" on Qwen3ForCausalLM would pass
# every other check, and the app's template matcher would read it as Qwen3 too).
FAMILY_NAME_PATTERNS = {
    "qwen3": re.compile(r"qwen3(?![.\d])", re.IGNORECASE),
    "qwen2_5": re.compile(r"qwen2\.5(?![.\d])", re.IGNORECASE),
    "lfm2": re.compile(r"lfm2\.5(?![.\d])", re.IGNORECASE),
    "llama": re.compile(r"llama-?3\.2(?![.\d])|smollm2(?!\d)", re.IGNORECASE),
    "gemma3": re.compile(r"gemma-?3(?![.\dn])", re.IGNORECASE),
    "smollm3": re.compile(r"smollm3(?!\d)", re.IGNORECASE),
}
_INSTRUCT_TOKENS = {"instruct", "it", "chat"}
# Families whose chat models ship with no suffix at all and whose base models carry "-Base".
# Qwen3 does this throughout. LFM2.5 does it unevenly: the 1.2B is LFM2.5-1.2B-Instruct
# against LFM2.5-1.2B-Base, but the 2.6B is plain LFM2.5-2.6B against LFM2.5-2.6B-Base, so
# reading the bare name as a base model gets that one backwards.
_CHAT_WITHOUT_A_SUFFIX = {"qwen3", "lfm2"}


@dataclass
class Verdict:
    model_id: str
    sha: str
    family: str | None
    variant: str
    # Reasons the model is skipped entirely; empty when it is eligible.
    reasons: list[str] = field(default_factory=list)
    # Backend → None when it will be exported, else why not.
    backends: dict[str, str | None] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return not self.reasons and any(v is None for v in self.backends.values())

    @property
    def export_backends(self) -> list[str]:
        return [b for b, why in self.backends.items() if why is None] if not self.reasons else []

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "sha": self.sha,
            "family": self.family,
            "variant": self.variant,
            "eligible": self.eligible,
            "reasons": self.reasons,
            "backends": self.backends,
        }


def variant(model_id: str, family: str | None) -> str:
    tokens = set(re.split(r"[-_.]", naming.source_name(model_id).lower()))
    if tokens & _INSTRUCT_TOKENS:
        return "instruct"
    if family in _CHAT_WITHOUT_A_SUFFIX and "base" not in tokens:
        return "instruct"
    return "base"


def name_reasons(model_id: str, settings: Settings) -> list[str]:
    """Reasons that need nothing but the repo name, so the watcher can skip a model before
    fetching anything about it."""
    name = naming.source_name(model_id)
    reasons = []
    hits = [token for token in settings.name_exclude if token in naming.normalise(name)]
    if hits:
        reasons.append(f"name matches excluded marker(s) {hits}")
    hints = naming.size_hints(name)
    nominal = naming.nominal_billions(name)
    if nominal is None:
        reasons.append(f"the app reads the size from the name and {name!r} has none (like 1.7B)")
    elif len(hints) > 1:
        reasons.append(f"name carries several sizes {hints}; the app would show the first")
    elif nominal > settings.max_nominal_billions:
        reasons.append(f"named size {nominal:g}B is above {settings.max_nominal_billions:g}B")
    output_repo = naming.output_repo(model_id, settings.hub_org, settings.repo_suffix)
    if naming.app_family(naming.app_model_name(output_repo, "x.pte")) is None:
        reasons.append(f"the app has no chat template for {name!r} (or refuses it by name)")
    return reasons


def evaluate(source: SourceModel, settings: Settings) -> Verdict:
    name = naming.source_name(source.id)
    family = families.family_for(source.config) if source.config else None
    family_key = family.key if family else None
    verdict = Verdict(source.id, source.sha, family_key, variant(source.id, family_key))
    reasons = verdict.reasons

    if source.pipeline_tag not in (None, settings.pipeline_tag):
        reasons.append(f"pipeline tag {source.pipeline_tag!r} is not {settings.pipeline_tag!r}")
    if verdict.variant == "base":
        # Base checkpoints are not published. They have no chat template, so the app -- which
        # renders one from the name and refuses a file it cannot place -- cannot run them,
        # and a completion model in a chat app reads as a broken chat model.
        reasons.append(f"{name!r} is a base checkpoint, and only chat models are published")
    reasons.extend(name_reasons(source.id, settings))
    if source.access_error:
        reasons.append(source.access_error)
    elif not source.config:
        reasons.append("repo has no config.json")
    elif families.is_moe(source.config):
        reasons.append("mixture-of-experts checkpoint")
    elif family is None:
        reasons.append(f"architecture {source.config.get('architectures')} has no recipe")
    if source.total_params is None:
        reasons.append("no safetensors parameter count on the Hub")
    elif source.total_params >= settings.max_params:
        reasons.append(f"{source.total_params:,} parameters is not below {settings.max_params:,}")

    output_repo = naming.output_repo(source.id, settings.hub_org, settings.repo_suffix)
    app_token = naming.app_family(naming.app_model_name(output_repo, "x.pte"))
    if family is not None and app_token is None:
        # The app reads the chat template from the name, so a file it cannot place is a file
        # it refuses to load after the download. LFM2 without the .5 is the live case.
        reasons.append(f"{source.id!r} reads as no app family, so the app could not render it")
    elif family is not None and app_token is not None:
        if app_token not in FAMILY_APP_TOKENS.get(family.key, set()):
            reasons.append(f"name reads as app family {app_token!r}, architecture is {family.key!r}")
        elif not FAMILY_NAME_PATTERNS[family.key].search(name):
            reasons.append(f"{name!r} is not named like a {family.key} release (a later version?)")

    for backend in families.BACKENDS:
        if family is None:
            verdict.backends[backend] = "no family"
        elif not family.supports(backend):
            verdict.backends[backend] = family.unsupported[backend]
        elif backend in ("xnnpack", "vulkan"):  # the same export_llm path, one delegate each
            try:
                families.xnnpack_plan(family, source.config)
                verdict.backends[backend] = None
            except families.UnsupportedModel as error:
                verdict.backends[backend] = str(error)
        elif backend == "qnn":
            verdict.backends[backend] = None if families.qnn_decoder(source.id) else families.QNN_UNLISTED
        elif backend == "mtk":
            try:
                families.mtk_plan(family, source.config, settings.mtk.max_chunks)
                verdict.backends[backend] = None
            except families.UnsupportedModel as error:
                verdict.backends[backend] = str(error)
        else:
            raise ValueError(f"unknown backend {backend!r}")
    return verdict
