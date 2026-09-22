"""Does the exported file still decide like the model it came from?

The smoke test proves a file generates: it loads through the same C++ runner the app uses
and answers a question. That is coherence, and it is necessary. It is not sufficient, and
one build proved it. A 1.2B export passed the smoke test, answered the capital of France
correctly, came out byte-identical in size to the file it would have replaced -- and had
quietly lost a tool call, because the int4 codes had been solved against a calibration set
with no tool schema in it. Nothing in the pipeline could see that, so it nearly shipped.

What this adds is the measurement that did see it. On prompts shaped like the app's, the
unquantised model is asked what it would do; then the exported file is asked the same thing
through the runner, and it has to agree. Two numbers, both from the research note's section
6 gate:

* it must still pick fp32's token on at least ``MIN_AGREEMENT`` of the rows fp32 is
  confident about;
* its mean probability on that token must not fall more than ``MAX_SHORTFALL`` below fp32's.

Both thresholds are set from four real files measured on these prompts, not chosen:

    file                                    mean on fp32's token   gap    kept
    round to nearest, state fixed (bad)              0.049     0.772 under   0/12
    GPTQ, calibrated without a tool schema           0.730     0.092 under  12/12
    GPTQ v2, the published file                      0.701     0.120 under  12/12
    GPTQ on an abliterated checkpoint                0.945     0.123 over   12/12

The file that lost its tool calling in the field -- 10 percent recall on the phone against
49 for a solved one -- keeps none of fp32's choices and sits three quarters of the way to
zero. Every file that works keeps all twelve. So "kept" is the signal and the mean is the
backstop, and the mean's band is one-sided: a file above fp32 is a different phenomenon
(abliteration removes the hesitation, and that export calls more readily than the model it
came from) and not something a publish gate should refuse.

Family-generic on purpose: no tool token is named anywhere here. On a trailered row the
token fp32 puts first *is* the tool call, so "keep fp32's top token" measures tool calling
without this file knowing what a tool call looks like in any particular family's markup.

The reference is computed in the solve job, where the fp32 model is already in memory, and
travels to the verify job with the codes. Without it -- a run with GPTQ off -- there is
nothing to compare against and the gate says so rather than passing quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

# How far the export's mean may fall below fp32's on the calling rows. The research note's
# section 6 gate was 0.05, but that was read against the lab's own probe set, which every
# recipe there was tuned against; on a fresh draw the published file itself is 0.120 under,
# so 0.05 refuses what is already shipping. 0.25 is twice the worst working file and a third
# of the broken one.
MAX_SHORTFALL = 0.25
# Of those rows, how many the export must still choose the same token on. This is the real
# signal: every working file keeps 12 of 12 and the broken one keeps none.
MIN_AGREEMENT = 0.9
# Rows where fp32 itself is undecided say nothing about a regression, so they are not graded.
CONFIDENT = 0.5

# Questions the calibration never sees: a gate drawn from the calibration set would grade a
# solve on its own homework. tests/test_gate.py holds the two apart.
SEARCH: tuple[tuple[str, str], ...] = (
    ("What is Concord the capital of?", "Concord"),
    ("What is Providence the capital of?", "Providence"),
    ("What is Augusta the capital of?", "Augusta"),
    ("What is Nassau the capital of?", "Nassau"),
    ("Who is the author of Mercy?", "author of Mercy"),
    ("Who is the author of Landfall?", "author of Landfall"),
    ("Who is the author of Quicksand?", "author of Quicksand"),
    ("Who is the author of Nightfall?", "author of Nightfall"),
    ("In what country is Ostrava?", "Ostrava"),
    ("In what country is Gdynia?", "Gdynia"),
    ("In what country is Rijeka?", "Rijeka"),
    ("In what country is Linkoping?", "Linkoping"),
)

# The other half of the decision: under the same app prompt, questions the model should
# answer itself. A file that started calling on these would be regressed too.
QUIET: tuple[str, ...] = (
    "What is the capital of Norway?",
    "What is 8 times 7?",
    "Who wrote Pride and Prejudice?",
    "What is the chemical symbol for iron?",
)


def prompts(tokenizer) -> dict[str, str]:
    """Every gate question in the app's own prompt, ready for the assistant's turn."""
    from pipeline import calibration

    out = {q: calibration.render_app(tokenizer, f"{q}\n\n{calibration.TRAILER.format(subject=s)}") for q, s in SEARCH}
    out.update({q: calibration.render_app(tokenizer, q) for q in QUIET})
    return out


def reference(model, tokenizer) -> dict:
    """What the unquantised model would do: its top token per prompt, and how sure it is."""
    import torch

    rows = {}
    for question, text in prompts(tokenizer).items():
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        ids = ids.to(next(model.parameters()).device)
        with torch.no_grad():
            logits = model(ids).logits
        probs = torch.softmax(logits[0, -1].float(), -1)
        token = int(probs.argmax())
        rows[question] = {"token": token, "p": float(probs[token]), "text": tokenizer.decode([token])}
    return {"rows": rows, "confident": sum(1 for r in rows.values() if r["p"] > CONFIDENT)}


def measure(pte: Path, tokenizer_path: Path, tokenizer, ref: dict) -> dict:
    """The same question put to the exported file, through the runner the app uses."""
    import tempfile
    import types

    import torch
    from executorch.examples.models.llama.runner.native import NativeLlamaRunner

    params = Path(tempfile.mkdtemp()) / "params.json"
    params.write_text(json.dumps({"vocab_size": len(tokenizer)}))
    rows = {}
    for question, text in prompts(tokenizer).items():
        # Reloaded per prompt: an LFM2 export carries convolution state, and a row that
        # started on the previous row's state would not be measuring this prompt.
        runner = NativeLlamaRunner(
            types.SimpleNamespace(
                pte=str(pte),
                params=str(params),
                tokenizer=str(tokenizer_path),
                tokenizer_config=None,
                max_len=2048,
                kv_cache=True,
            )
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        logits = runner.forward(torch.tensor([ids], dtype=torch.long), torch.tensor([0], dtype=torch.long))
        last = (logits[0, -1] if logits.dim() == 3 else logits[-1]).float()
        probs = torch.softmax(last, -1)
        want = ref["rows"][question]["token"]
        rows[question] = {"p": float(probs[want]), "top": int(probs.argmax())}
    return rows


def check(ref: dict, measured: dict) -> dict:
    """Grade one export against its own fp32 model. ``passed`` is what blocks a publish.

    The two kinds of row are graded apart, not pooled. Pooling them was a real mistake on
    the first version of this file: a quiet row where fp32 answers "The" at 0.99 is tracked
    perfectly by any export that works at all, so mixing those into the mean pulled it
    toward 1.0 and buried the only rows that carry the signal. The lab's gate kept them
    separate for the same reason -- a mean over the calling rows, and a separate count of
    calls made where fp32 would not have called.
    """
    search = [q for q, _ in SEARCH if ref["rows"][q]["p"] > CONFIDENT]
    undecided = [q for q, _ in SEARCH if ref["rows"][q]["p"] <= CONFIDENT]
    if not search:
        return {"passed": False, "problems": ["fp32 called on no gate row at all"], "graded": 0}

    fp32_mean = sum(ref["rows"][q]["p"] for q in search) / len(search)
    export_mean = sum(measured[q]["p"] for q in search) / len(search)
    agreed = sum(1 for q in search if measured[q]["top"] == ref["rows"][q]["token"])
    # A quiet row is one where fp32 answers instead of calling. The export must do the same:
    # picking fp32's token there means it answered, and any other top token is worth seeing.
    quiet_followed = sum(1 for q in QUIET if measured[q]["top"] == ref["rows"][q]["token"])
    # Whether this model tool-calls at all. LFM2.5-2.6B answers "The" at probability 1.0 to
    # every row here, trailered or not: it is not tool-trained, so fp32's chosen token is the
    # same on the search rows as on the quiet ones and the gate cannot see a tool-calling
    # regression in it. That is not a failure -- the export still has to track fp32, and it
    # does -- but a verdict that said nothing about it would be read as if it had.
    quiet_tokens = {ref["rows"][q]["token"] for q in QUIET}
    calls = sum(1 for q in search if ref["rows"][q]["token"] not in quiet_tokens)
    informative = calls >= len(search) // 2

    problems = []
    if fp32_mean - export_mean > MAX_SHORTFALL:
        problems.append(
            f"mean probability on fp32's token over {len(search)} calling rows is {export_mean:.3f} "
            f"against fp32's {fp32_mean:.3f}, {fp32_mean - export_mean:.3f} under and the gate is {MAX_SHORTFALL}"
        )
    if agreed < MIN_AGREEMENT * len(search):
        problems.append(f"kept fp32's choice on {agreed} of {len(search)} calling rows, under {MIN_AGREEMENT:.0%}")
    if quiet_followed < len(QUIET) - 1:
        problems.append(
            f"answered like fp32 on only {quiet_followed} of {len(QUIET)} quiet rows: it is reaching for a tool "
            "on questions the unquantised model answers itself"
        )
    return {
        "passed": not problems,
        "problems": problems,
        "graded": len(search),
        "fp32_mean": round(fp32_mean, 4),
        "export_mean": round(export_mean, 4),
        "agreed": agreed,
        "shortfall": round(fp32_mean - export_mean, 4),
        "quiet_rows_answered_like_fp32": quiet_followed,
        "measures_tool_calling": informative,
        "quiet_rows": len(QUIET),
        "fp32_undecided_rows": undecided,
        "rows": {
            q: {"fp32": round(ref["rows"][q]["p"], 4), "export": round(measured[q]["p"], 4), "kind": kind}
            for kind, group in (("search", [q for q, _ in SEARCH]), ("quiet", list(QUIET)))
            for q in group
        },
    }


def measuring_gate(stats: dict | None = None) -> dict:
    """A non-blocking gate measuring prefill and decode throughput without failing push."""
    if not stats:
        return {
            "kind": "measuring_gate",
            "passed": True,
            "prefill_tok_per_sec": None,
            "decode_tok_per_sec": None,
            "prompt_tokens": None,
            "generated_tokens": None,
        }
    prefill = stats.get("prefill_token_per_sec")
    decode = stats.get("decode_token_per_sec")
    prompt_tok = stats.get("prompt_tokens") or stats.get("num_prompt_tokens")
    gen_tok = stats.get("generated_tokens") or stats.get("num_generated_tokens")
    return {
        "kind": "measuring_gate",
        "passed": True,
        "prefill_tok_per_sec": round(float(prefill), 2) if prefill is not None else None,
        "decode_tok_per_sec": round(float(decode), 2) if decode is not None else None,
        "prompt_tokens": int(prompt_tok) if prompt_tok is not None else None,
        "generated_tokens": int(gen_tok) if gen_tok is not None else None,
    }
