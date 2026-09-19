"""Smoke-test prompts, rendered the way the app feeds the runtime.

The 1.4.0 runtime never adds BOS, so the app writes the family's BOS as text and relies on
the tokenizer to encode it (openweights PromptTemplate.kt). Rendering the upstream chat
template with ``bos_token`` filled in produces the same kind of prompt.

Two things here were found the hard way on 2026-09-19, by an abliterated LFM2.5 that failed
every window of its export and turned out not to be broken at all.

*Render through transformers when it is installed.* LFM2.5's chat template uses ``{%
generation %}``, which is transformers' own Jinja extension and which the sandbox below
cannot compile. Every LFM2.5 export ever smoke-tested here therefore fell back to the plain
completion prompt and the chat template was never exercised -- silently, because the
fallback answers the question well enough to pass. transformers is already a dependency of
the export, so its renderer is used first and the sandbox is the fallback, not the rule.

*Never drop BOS.* The fallback used to write BOS only when ``add_bos_token`` was set.
transformers 5.x does not write that key, so a checkpoint re-saved by it got a prompt with
no BOS -- and LFM2.5 without BOS answers " is is is is is" to anything, which reads exactly
like a destroyed export. Measured both ways on both checkpoints: with BOS both answer
"Paris. It is the most populous city in France"; without it both degenerate. The app always
writes BOS, so the smoke test always writes BOS.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

QUESTION = "What is the capital of France? Answer with one word."
COMPLETION = "The capital of France is"
EXPECTED = "paris"


def _template(model_dir: Path, tokenizer_config: dict) -> str | None:
    standalone = model_dir / "chat_template.jinja"
    if standalone.exists():
        return standalone.read_text(encoding="utf-8")
    template = tokenizer_config.get("chat_template")
    if isinstance(template, list):  # named templates: use the default one
        by_name = {t.get("name"): t.get("template") for t in template}
        template = by_name.get("default") or next(iter(by_name.values()), None)
    return template


def _token_text(value) -> str:
    if isinstance(value, dict):
        return value.get("content", "")
    return value or ""


def _environment():
    # The environment transformers renders chat templates in.
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def _render_with_transformers(model_dir: Path) -> str | None:
    """The tokenizer's own chat template, which knows the extensions the sandbox does not."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": QUESTION}], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return None
    return text if isinstance(text, str) and text.strip() else None


def render(model_dir: Path, tokenizer_config: dict, instruct: bool) -> str:
    """The prompt string for the smoke test."""
    bos = _token_text(tokenizer_config.get("bos_token"))
    if instruct:
        rendered = _render_with_transformers(model_dir)
        if rendered is not None:
            # apply_chat_template may or may not have written BOS itself; the runtime adds
            # none, so it has to be in the text exactly once. The test is whether the token
            # is present, not whether it is first: a template that opens with a newline and
            # then {{ bos_token }} renders BOS in second place, and asking startswith() put
            # a second one in front of it (Codex review, 2026-09-19). Two BOS tokens is a
            # sequence the model never saw in training, which degrades it quietly -- the
            # same shape of fault as the missing BOS this function was fixed for.
            return rendered if not bos or bos in rendered else bos + rendered
    template = _template(model_dir, tokenizer_config) if instruct else None
    if template is None:
        # BOS unconditionally: see the module docstring. Gating it on add_bos_token cost a
        # healthy abliterated model five failed windows.
        return bos + COMPLETION
    variables = {
        "messages": [{"role": "user", "content": QUESTION}],
        "add_generation_prompt": True,
        "bos_token": bos,
        "eos_token": _token_text(tokenizer_config.get("eos_token")),
        "enable_thinking": False,
    }
    return _environment().from_string(template).render(**variables)
