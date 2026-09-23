"""The samples a MediaTek LLM export calibrates on, sized to the window it is built for.

MediaTek's recipe calibrates on `alpaca.txt`, nine prompts of 4 to 42 words. The quantizer's
activation observers are MinMax (backends/mediatek/quantizer/qconfig.py), so every 16-bit
activation range, the cache's included, is whatever those nine prompts produced at positions
under about sixty. A 4k or 32k window then runs positions the observers never saw: RoPE's slow
channels rotate through values a sixty-token prompt cannot reach, and a value outside the
observed range is clipped. The 4-bit weights are not affected (per-channel MinMax over the
weights themselves); the activations and the cache are.

So the samples here fill the cache, from a few hundred tokens to the whole window, with what a
phone actually sends: long conversations under the app's own system prompt, tools and opening
exchange, and long pasted documents with a question after them. MediaTek's nine prompts stay in
as the short samples, so every range is at least what the old recipe measured. Measured on
LFM2.5-1.2B at a 2k window: the long samples alone widened 327 of 391 live ranges (median 1.09x,
up to 3.25x), alpaca's range missed over a quarter of the long-context range in 75 of them, and
12 were wider under alpaca, which is why it stays. What that is worth on the output is small:
at 4k, KL to fp32 went from 0.751 to 0.745 on held-out text, because the 4-bit weights are the
error (A16W8 on the same text: 0.0068; docs/research finding 35).

Every sample is rendered in the model's own chat template, which is also what gives it exactly
one BOS: the old recipe wrapped alpaca in Qwen3's template, which has none, so a model whose
tokenizer does not add BOS by itself (LFM2.5-2.6B) was calibrated without one.

The sources are pinned by revision and sha256, so a run is repeatable:

- UltraChat 200k (HuggingFaceH4/ultrachat_200k, MIT), its `test_sft` split, for conversations.
- PG19 (deepmind/pg19, Apache-2.0; the books are public domain), seven books of its test split,
  from the bucket the dataset's own loader reads, for documents.

Two halves, because they run in different interpreters: `fetch` and `plan` are standard library
and run in the pipeline's Python; `build` needs the tokenizer and pyarrow and runs in MediaTek's
toolchain virtualenv, which has both (`python -m pipeline.mtk_corpus`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ULTRACHAT = {
    "repo_id": "HuggingFaceH4/ultrachat_200k",
    "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "filename": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
    "sha256": "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70",
    "license": "MIT",
}
PG19_URL = "https://storage.googleapis.com/deepmind-gutenberg/test/{id}.txt"
# (Gutenberg id, title, sha256). Each is long enough for a 32k window on its own.
PG19_BOOKS = (
    (
        "10146",
        "Reminiscences of Pioneer Days in St. Paul, Frank Moore (1908)",
        "f8dd4334fd8d422cd2f9153ffbff9b097611569592d7ca09fc6fe7881336d331",
    ),
    (
        "10321",
        "Dragon's Blood, Henry Milner Rideout (1909)",
        "fedcb2723c2ad091f76d45eb98efd810977e7b906f6fd8cbd815734d1e90c456",
    ),
    (
        "10356",
        "Travels in Morocco, Volume 2, James Richardson (1860)",
        "af4b57d358448588adbc6da962cad655bf586ae746dba21f62e956d36d97c88e",
    ),
    (
        "10762",
        "Impressions of Theophrastus Such, George Eliot (1879)",
        "fcc71b161311d4854a078f8cdda6e4381542536167145f6dfdda84bcad256d5f",
    ),
    (
        "15562",
        "The S. W. F. Club, Caroline E. Jacobs (1912)",
        "90eae481a404688312d407918b07e649643ce6c934a16ad31040614221673ca1",
    ),
    (
        "2544",
        "From Sand Hill to Pine, Bret Harte (1900)",
        "0c79ceaf5bcc208b8c924037613c2c7fb1606de21ec8d99d5c446a516c4af787",
    ),
    (
        "22424",
        "Frank Merriwell Down South, Burt L. Standish (1903)",
        "a3475140ed4763614c11c4f2e486e468ae5620bf539877bf769e16710d808a0d",
    ),
)
PG19_LICENSE = "Apache-2.0 (PG19); the books are public domain"

# What follows a pasted passage. Written for this file, so no graded question reaches a solve.
DOCUMENT_ASKS = (
    "Summarise the passage above in three sentences.",
    "Who are the main people in this passage, and what does each of them want?",
    "List the places the passage above mentions, one per line.",
    "What happens at the end of this excerpt?",
    "Pick the sentence from the passage that best shows its tone, quote it, and say why.",
    "Is there anything in the passage above that a modern reader would find surprising? Explain.",
    "Write a one-line title for this passage.",
)
# A sample that would hold the app's prefix needs room for more than the prefix.
PREFIX_ROOM = 512


def plan(window: int, response_cap: int, samples: int, short: int) -> list[tuple[str, int]]:
    """(kind, target prompt tokens) per sample: the short prompts, then `samples` long ones.

    The prompt and every generated step share the cache, so no prompt may be longer than the
    window less the steps after it. The long samples are spread evenly up to that limit, so
    every position the window can hold is observed, near the start by every sample and near
    the end by the longest; chat and document alternate so both kinds reach the far end.
    """
    if samples < 1:
        raise ValueError("at least one long sample, the one that fills the window")
    limit = window - response_cap - 1
    floor = min(256, limit)
    out = [("short", 0) for _ in range(short)]
    for i in range(samples):
        target = round(floor + (limit - floor) * (i + 1) / samples)
        out.append(("chat" if i % 2 == 0 else "document", target))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(dest: Path) -> dict[str, Path]:
    """Download the pinned sources into dest (idempotent) and check every byte."""
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    chat = Path(
        hf_hub_download(
            ULTRACHAT["repo_id"],
            ULTRACHAT["filename"],
            repo_type="dataset",
            revision=ULTRACHAT["revision"],
            local_dir=dest / "ultrachat",
        )
    )
    if _sha256(chat) != ULTRACHAT["sha256"]:
        raise RuntimeError(f"{chat} does not match the pinned sha256")
    paths["ultrachat"] = chat
    books = dest / "pg19"
    books.mkdir(exist_ok=True)
    for book_id, _, digest in PG19_BOOKS:
        path = books / f"{book_id}.txt"
        if not path.exists() or _sha256(path) != digest:
            with urllib.request.urlopen(PG19_URL.format(id=book_id), timeout=60) as r:
                path.write_bytes(r.read())
        if _sha256(path) != digest:
            raise RuntimeError(f"PG19 book {book_id} does not match the pinned sha256")
    paths["pg19"] = books
    return paths


def describe(summary: dict) -> str:
    """One sentence for the model card and the export report."""
    kinds = summary["kinds"]
    return (
        f"{summary['samples']} samples in the model's own chat template filling the cache from "
        f"{summary['fill_min']:,} to {summary['fill_max']:,} tokens ({kinds.get('chat', 0)} "
        f"conversations from UltraChat 200k, under the app's system prompt where it fits, "
        f"{kinds.get('document', 0)} passages from PG19 books, and MediaTek's {kinds.get('short', 0)} "
        f"`alpaca.txt` prompts), "
        f"{summary['tokens_total']:,} tokens observed"
    )


# Everything below runs in MediaTek's toolchain virtualenv.


def _tokenizer(weights: Path):
    """The source's tokenizer with its chat template, loaded the way lfm2.py loads it."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
    tok_config = json.loads((weights / "tokenizer_config.json").read_text(encoding="utf-8"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(weights)
    except Exception:
        # transformers 4.57 in this virtualenv does not know TokenizersBackend, the class a
        # transformers-5 tokenizer_config names; lfm2.py falls back the same way.
        def token(name):
            value = tok_config.get(name)
            return value.get("content") if isinstance(value, dict) else value

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(weights / "tokenizer.json"),
            bos_token=token("bos_token"),
            eos_token=token("eos_token"),
            pad_token=token("pad_token"),
        )
    if not getattr(tokenizer, "chat_template", None):
        jinja = weights / "chat_template.jinja"
        tokenizer.chat_template = jinja.read_text(encoding="utf-8") if jinja.exists() else tok_config["chat_template"]
    return tokenizer, config.get("bos_token_id")


class _Renderer:
    def __init__(self, tokenizer, bos_id):
        self.tokenizer = tokenizer
        self.bos_id = bos_id

    def ids(self, messages, tools=None) -> list[int]:
        text = self.tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        # Exactly one BOS, whichever of the template and the tokenizer is meant to write it.
        if self.bos_id is not None:
            while len(ids) > 1 and ids[0] == self.bos_id and ids[1] == self.bos_id:
                ids = ids[1:]
            if not ids or ids[0] != self.bos_id:
                ids = [self.bos_id, *ids]
        return ids


def _app_head():
    from pipeline import calibration

    return [{"role": "system", "content": calibration.APP_SYSTEM}, *calibration.APP_PRIMER], calibration.APP_TOOLS


def _pairs(parquet: Path):
    """(user, assistant) exchanges, conversation after conversation, in file order."""
    import pyarrow.parquet as pq

    for row in pq.read_table(parquet, columns=["messages"]).column("messages").to_pylist():
        turns = [m for m in row if m["role"] in ("user", "assistant") and m["content"].strip()]
        for i in range(0, len(turns) - 1, 2):
            if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
                yield turns[i], turns[i + 1]


def _chat(render: _Renderer, target: int, head, tools, pairs) -> list[int]:
    """Earlier exchanges as history, closed by the user turn of the exchange that did not fit.

    Closing on a user turn is what a phone sends, and it means the reply the model writes
    during calibration answers something rather than continuing an exchange that had ended.
    """
    messages = list(head)
    pending = next(pairs)
    while len(render.ids(messages + [pending[0]], tools)) > target:
        pending = next(pairs)  # a question longer than the whole budget: take another
    while True:
        following = next(pairs)
        grown = messages + list(pending)
        if len(render.ids(grown + [following[0]], tools)) > target:
            break
        messages, pending = grown, following
    # One reply runs to hundreds of tokens, so whole exchanges leave most of a small budget
    # empty. The exchange that did not fit goes in with its reply cut to what is left.
    user, reply = pending
    reply_ids = render.tokenizer(reply["content"], add_special_tokens=False)["input_ids"]

    def with_reply(n):
        cut = {"role": "assistant", "content": render.tokenizer.decode(reply_ids[:n])}
        return render.ids(messages + [user, cut, following[0]], tools)

    lo, hi = 0, len(reply_ids)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(with_reply(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    if lo >= 32:
        return with_reply(lo)
    return render.ids(messages + [user], tools)


def _document(render: _Renderer, target: int, head, tools, book_ids, ask, tokenizer) -> list[int]:
    def ids_for(n):
        excerpt = tokenizer.decode(book_ids[:n])
        content = f"Here is a passage from a book.\n\n{excerpt.strip()}\n\n{ask}"
        return render.ids(head + [{"role": "user", "content": content}], tools)

    overhead = len(ids_for(0))
    n = max(0, target - overhead)
    ids = ids_for(n)
    # Decoding and re-encoding a slice is not exactly length preserving; close in from above.
    while len(ids) > target and n > 0:
        n -= len(ids) - target + 4
        ids = ids_for(max(n, 0))
    return ids


def build(
    weights: Path,
    sources: dict[str, Path],
    short_prompts: list[str],
    window: int,
    response_cap: int,
    samples: int,
    out: Path,
) -> dict:
    tokenizer, bos_id = _tokenizer(weights)
    render = _Renderer(tokenizer, bos_id)
    app_messages, app_tools = _app_head()
    prefix = len(render.ids(app_messages, app_tools))
    stream = _pairs(sources["ultrachat"])
    books = []
    for book_id, title, _ in PG19_BOOKS:
        text = (sources["pg19"] / f"{book_id}.txt").read_text(encoding="utf-8").replace("\r\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        # Skip the front matter (producers, title page), which is not what anyone pastes.
        books.append((title, tokenizer(text[len(text) // 20 :], add_special_tokens=False)["input_ids"]))

    rows = []
    documents = 0
    for kind, target in plan(window, response_cap, samples, len(short_prompts)):
        if kind == "short":
            ids = render.ids([{"role": "user", "content": short_prompts[sum(r["kind"] == "short" for r in rows)]}])
        else:
            with_prefix = target >= prefix + PREFIX_ROOM
            head = app_messages if with_prefix else []
            tools = app_tools if with_prefix else None
            if kind == "chat":
                ids = _chat(render, target, head, tools, stream)
            else:
                _title, book_ids = books[documents % len(books)]
                ask = DOCUMENT_ASKS[documents % len(DOCUMENT_ASKS)]
                ids = _document(render, target, head, tools, book_ids, ask, tokenizer)
                documents += 1
        if len(ids) + response_cap >= window:
            raise RuntimeError(f"a {kind} sample of {len(ids)} tokens leaves no room for {response_cap} replies")
        rows.append({"kind": kind, "tokens": len(ids), "input_ids": ids})

    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    lengths = [r["tokens"] for r in rows]
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {
        "corpus": "long-context-v1",
        "window": window,
        "samples": len(rows),
        "kinds": kinds,
        "tokens_total": sum(lengths) + len(rows) * response_cap,
        "fill_min": min(lengths),
        "fill_max": max(lengths),
        "lengths": lengths,
        "app_prefix_tokens": prefix,
        "sources": {
            "ultrachat": {k: ULTRACHAT[k] for k in ("repo_id", "revision", "filename", "sha256", "license")},
            "pg19": {"books": [{"id": b, "title": t, "sha256": d} for b, t, d in PG19_BOOKS], "license": PG19_LICENSE},
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True, help="the directory fetch() filled")
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--response-cap", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True, help="long samples, besides the short prompts")
    parser.add_argument("--short-prompts", type=Path, required=True, help="one prompt per line (MediaTek's alpaca.txt)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = {"ultrachat": args.sources / "ultrachat" / ULTRACHAT["filename"], "pg19": args.sources / "pg19"}
    short = [line for line in args.short_prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = build(args.weights, sources, short, args.window, args.response_cap, args.samples, args.out)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"calibration corpus: {describe(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
