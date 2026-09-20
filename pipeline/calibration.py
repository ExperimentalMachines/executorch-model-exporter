"""The text GPTQ takes its Hessians from.

A calibration set decides what the solve optimises for, so it is committed here rather than
fetched: a run must be repeatable, and a corpus that changes underneath the pipeline would
change published weights without changing any code.

Three decisions, each measured rather than assumed.

*Rendered in the model's own chat template.* The exports are chat models and the app that
runs them always sends a template, so calibrating on raw text would take the Hessians from a
token distribution the model never meets in use.

*Continued by the model itself.* Each prompt is followed by the fp32 model's own greedy
reply before the sequence is used. What GPTQ protects is the output the unquantised model
would have produced, and the reply is where most of those positions are; calibrating on
prompts alone leaves the decision to answer, or to call a tool, almost unrepresented.

*Half the rows wear the app's own prompt: its system text, its tool schema and its opening
exchange.* This is the ingredient that decides tool calling, and the reason it is here is a
measurement. The lab built two solves for LFM2.5-1.2B. The first (v1) calibrated on the
app's prompt alone and tilted toward it: on the app's rows its mean tool-call probability
was 0.871 against fp32's 0.756, and on held-out benchmark rows its KL to fp32 was 0.211,
barely better than round to nearest's 0.307. The second (v2) kept the app's prompt and
added a held-out GSM8K, IFEval and BFCL draw -- 277 rows, 35,804 counted positions -- and
came back three to four times closer to fp32 on general prompts (benchmark KL 0.074) while
still calling: 0.743 on the app's calling rows, 15 of 16 calls kept, no call fp32 would not
make. On the phone the two were inside the suite's noise of each other (searched when
needed 51 against 48 percent over 75 rows). v2 is what shipped, and it is what this file
reproduces. Sources: docs/research/executorch-state-and-recipes.md section 6 in the
openweights repository, and tools/executorch/quantlab/calib.py, which built both.

What is *not* copied from the lab is the questions. The lab drew its app rows from a graded
seed-8 set and filtered the test sets out row by row; the questions below are written for
this file instead, so no graded question can reach a solve through this repository at all.

The shared head is counted once. Every app-shaped row opens with the same ~600 tokens, and
counting them in every row would let the head dominate the Hessian and drown the questions
and replies, which are the positions the decision actually lives in. ``sequences`` returns a
``keep_from`` per row and ``solve.solve_model`` accumulates from there, which is what
``quantlab/gptq.py`` does ("Rows that share a file's head count it once").
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Kept short on purpose: every prompt costs an fp32 generation pass on the runner, and the
# Hessians converge long before the corpus is large. Thirty-two rows at up to 256 tokens is
# a few thousand positions per linear, which is the range GPTQ's own paper calibrates in.
PROMPTS: tuple[str, ...] = (
    # Plain knowledge, short answers.
    "What is the capital of France?",
    "Who wrote the novel Frankenstein?",
    "What is the boiling point of water at sea level, in Celsius?",
    "Name the largest moon of Saturn.",
    "In which year did the Berlin Wall come down?",
    "What language is spoken in Brazil?",
    # Knowledge the model should decline or hedge on, so refusal and uncertainty are in the data.
    "What is the population of the village of Locenice right now?",
    "Who won the football match last night?",
    "What is my bank balance?",
    # Instruction following with a format constraint.
    "List three primary colours, one per line, with no other text.",
    "Rewrite this sentence in the passive voice: The committee approved the proposal.",
    "Summarise in exactly one sentence: a long meeting produced no decisions.",
    "Reply with only the word OK.",
    # Reasoning and arithmetic.
    "A shop sells pens at 3 for 2.40. What do five pens cost?",
    "If today is Wednesday, what day is it in 10 days?",
    "Sort these numbers in ascending order: 14, 3, 27, 8.",
    "A train leaves at 09:15 and arrives at 11:05. How long is the journey?",
    # Short code.
    "Write a Python function that returns the second largest number in a list.",
    "What does this do?\n\nfor i in range(3):\n    print(i * 2)",
    "Fix the bug: def mean(xs): return sum(xs) / len(xs) - 1",
    # Multi-turn shape, so the template's turn boundaries appear.
    "Thanks, that helps. Can you explain the last step again?",
    "No, I meant the other one.",
    # Longer context, so the solve sees prompts that are not all short.
    (
        "Here are some notes from a meeting. Decide what the action items are.\n\n"
        "Ana said the release slipped because the test runner was flaky on Windows. "
        "Ben offered to look at it on Thursday. Carla asked whether we should cut Windows "
        "support for this release; nobody answered. The deadline is the end of the month "
        "and the changelog has not been written."
    ),
    (
        "Explain, for someone who has never used one, what a compiler does and how it "
        "differs from an interpreter. Keep it under 150 words."
    ),
    # Tool-shaped prompts. The decision to call is the behaviour that quantisation loses
    # first on hybrid families, so it has to be represented in what the solve protects.
    "What is the weather in Manila right now?",
    "Search the web and tell me who won the 2026 Eurovision Song Contest.",
    "Look up the current price of a barrel of Brent crude.",
    "What time is it in Tokyo?",
    "Find me the opening hours of the British Library today.",
    # And the matching negatives: questions a model should answer from its own weights
    # rather than reaching for a tool, so the solve does not simply learn to always call.
    "What is 17 times 23?",
    "How many sides does a hexagon have?",
    "Translate 'good morning' into Spanish.",
)

# The app's own prompt, which every export in this repository is built to be run under.
# Copied from the prompt the openweights app sends (and from the probe prompts the lab
# measures tool calling on, where it is a byte-exact 2,431-character prefix once the
# tokenizer's chat template has rendered it).
# Read rather than inlined so the bytes stay exactly the app's: the text is one long
# paragraph per line and re-wrapping it here would change the tokens the solve calibrates on.
APP_SYSTEM = (Path(__file__).with_name("app_prompt.txt")).read_text(encoding="utf-8")

# Passed to ``apply_chat_template(tools=...)``, so each family renders it in its own markup
# rather than this file hard-coding one. Key order is load-bearing: the template serialises
# the dict as it is given, and for LFM2.5 this order reproduces the lab's head byte for byte.
APP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for what you cannot already know: what changed, what is recent, or the "
                "present state of a named person, product or organisation. Returns text; for pictures or "
                "clips use show_pictures. Not for settled knowledge (definitions, translations, history, "
                "arithmetic) and never to double check what you know: answer those yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up, as you would type it into a search box",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

# The app opens every conversation with the date and a ready exchange, so the questions the
# solve sees are never the first turn.
APP_PRIMER: tuple[dict[str, str], ...] = (
    {"role": "user", "content": "Today is 2026-09-10."},
    {
        "role": "assistant",
        "content": "Understood, I have that. I will not bring it up unless a question depends on it.",
    },
    {"role": "user", "content": "Ready when you are."},
    {"role": "assistant", "content": "Ready."},
)

# The app appends this to a question whose answer it wants fetched rather than recalled. It
# is what puts tool calls in the calibration replies without anyone labelling a row.
TRAILER = (
    "(This question names {subject}. Look it up with web_search before answering rather than "
    "recalling it, and answer from what the search returns.)"
)

# Questions that get the trailer, as (question, subject). Written for this file: the lab drew
# its app rows from a graded set and filtered the test rows out one by one, and none of the
# questions here appears in any of those files, so no graded question can reach a solve
# through this repository.
SEARCH_ROWS: tuple[tuple[str, str], ...] = (
    # Place names that are a capital somewhere and an ordinary town elsewhere.
    ("What is Springfield the capital of?", "Springfield"),
    ("What is Georgetown the capital of?", "Georgetown"),
    ("What is Hamilton the capital of?", "Hamilton"),
    ("What is Victoria the capital of?", "Victoria"),
    ("What is Salem the capital of?", "Salem"),
    ("What is Kingston the capital of?", "Kingston"),
    ("What is Albany the capital of?", "Albany"),
    ("What is Dover the capital of?", "Dover"),
    ("What is Richmond the capital of?", "Richmond"),
    ("What is Bridgetown the capital of?", "Bridgetown"),
    # One-word titles that belong to more than one book.
    ("Who is the author of Foundation?", "author of Foundation"),
    ("Who is the author of Wilderness?", "author of Wilderness"),
    ("Who is the author of Silence?", "author of Silence"),
    ("Who is the author of Harvest?", "author of Harvest"),
    ("Who is the author of Drift?", "author of Drift"),
    ("Who is the author of Thirst?", "author of Thirst"),
    ("Who is the author of Homeland?", "author of Homeland"),
    ("Who is the author of Threshold?", "author of Threshold"),
    ("Who is the author of Undercurrent?", "author of Undercurrent"),
    ("Who is the author of Passage?", "author of Passage"),
    # Places obscure enough that recalling them is the failure mode.
    ("In what country is Kryvyi Rih?", "Kryvyi Rih"),
    ("In what country is Zaanstad?", "Zaanstad"),
    ("In what country is Bielsko-Biala?", "Bielsko-Biala"),
    ("In what country is Trencin?", "Trencin"),
    ("In what country is Vasteras?", "Vasteras"),
    ("In what country is Kuopio?", "Kuopio"),
    ("In what country is Maribor?", "Maribor"),
    ("In what country is Pecs?", "Pecs"),
    ("In what country is Bydgoszcz?", "Bydgoszcz"),
    ("In what country is Tartu?", "Tartu"),
    # Institutions and organisations with names that do not say where they are.
    ("What is the Ruhr Museum known for?", "Ruhr Museum"),
    ("What does the Leibniz Institute of Freshwater Ecology do?", "Leibniz Institute of Freshwater Ecology"),
    ("What is the Slovak Philharmonic?", "Slovak Philharmonic"),
    ("What is the Tampere Hall used for?", "Tampere Hall"),
    ("What denomination is Emmanuel Reformed Church?", "Emmanuel Reformed Church"),
    ("What is the Bergen Maritime Museum?", "Bergen Maritime Museum"),
    # Live or recent state, which is the case the tool exists for.
    ("What is the current population of Tallinn?", "current population of Tallinn"),
    ("Who is the current mayor of Porto?", "current mayor of Porto"),
    ("What is the latest version of the Linux kernel?", "latest version of the Linux kernel"),
    ("Which team currently leads the Eredivisie?", "team currently leading the Eredivisie"),
    ("What is the current exchange rate between the euro and the zloty?", "current euro to zloty exchange rate"),
    ("Who currently holds the world record in the marathon?", "current marathon world record holder"),
    ("What is the most recent film directed by Lynne Ramsay?", "most recent film directed by Lynne Ramsay"),
    ("What is the current status of the Gotthard Base Tunnel?", "current status of the Gotthard Base Tunnel"),
    (
        "Which countries joined the European Union most recently?",
        "countries that joined the European Union most recently",
    ),
    ("What is the current price of a litre of diesel in Germany?", "current price of diesel in Germany"),
    ("Who won the most recent Booker Prize?", "most recent Booker Prize winner"),
    ("What is the current interest rate set by the Bank of England?", "current Bank of England interest rate"),
)

# Questions under the same app prompt that the model should answer from its own weights. A
# calibration made only of trailered rows would teach the solve that every question ends in
# a call, which is the tilt v1 had.
KNOWN_ROWS: tuple[str, ...] = (
    "What is the capital of Portugal?",
    "How many minutes are in three hours?",
    "What is the chemical symbol for potassium?",
    "Translate 'thank you' into Italian.",
    "What is the square root of 144?",
    "Who painted the Mona Lisa?",
    "How many continents are there?",
    "What is the past tense of 'bring'?",
    "What is 15 percent of 200?",
    "Name the four cardinal directions.",
    "What gas do plants absorb during photosynthesis?",
    "How many strings does a standard violin have?",
    "What is the largest planet in the solar system?",
    "Convert 25 degrees Celsius to Fahrenheit.",
    "What does the abbreviation 'etc.' stand for?",
    "Which ocean lies between Africa and Australia?",
    "What is the plural of 'crisis'?",
    "How many players are on a basketball team on court?",
    "What is the freezing point of water in Fahrenheit?",
    "Who wrote the play Hamlet?",
    "What is 9 times 12?",
    "What language is spoken in Austria?",
    "How many degrees are in a right angle?",
    "What is the currency of Japan?",
)

MAX_PROMPT_TOKENS = 512
MAX_REPLY_TOKENS = 128
# The lab's app rows took 48 (quantlab/calib.py --reply). The decision and the call's opening
# arguments are inside that; more reply on a ~600-token row buys positions at the highest
# price in the set.
APP_REPLY_TOKENS = 48


def render(tokenizer, prompt: str) -> str:
    """The prompt in the model's own chat template, ready for the assistant's turn."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def render_app(tokenizer, user_turn: str) -> str:
    """The same, under the app's system text, tool schema and opening exchange."""
    messages = [{"role": "system", "content": APP_SYSTEM}, *APP_PRIMER, {"role": "user", "content": user_turn}]
    return tokenizer.apply_chat_template(messages, tools=APP_TOOLS, tokenize=False, add_generation_prompt=True)


def app_head(tokenizer) -> str:
    """Everything the app-shaped rows share, up to where the question starts."""
    marker = "\x00question\x00"
    return render_app(tokenizer, marker).split(marker)[0]


def extended() -> dict[str, list[str]]:
    """The experiment's extra rows, or nothing.

    vLLM's LLM Compressor -- the reference implementation for production GPTQ -- calls for
    128 to 512 calibration sequences and says to start at 128; this file ships 104, which is
    under that floor. `config/calibration-extended.json` holds 273 further questions lifted
    from the lab corpus that the hand-built exports used, with every question that appears in
    the graded decision set removed (3 of them) and duplicates of this file's own questions
    removed (1). Set ``OW_CALIB_EXTENDED=1`` to include them, which is how the experiment is
    run; unset, this file behaves exactly as before and the published recipe does not move.
    """
    mode = os.environ.get("OW_CALIB_EXTENDED", "")
    if mode not in {"1", "full", "app"}:
        return {"app": [], "plain": []}
    path = Path(__file__).resolve().parent.parent / "config" / "calibration-extended.json"
    if not path.exists():
        raise FileNotFoundError(f"OW_CALIB_EXTENDED={mode} but {path} is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    extra = {"app": list(data.get("app") or []), "plain": list(data.get("plain") or [])}
    # "app" adds only the app-shaped rows. The 120 plain rows are GSM8K-style word problems,
    # and on Qwen3 -- a model that writes a <think> block before answering -- they turned a
    # 3.6x row increase into a 6x token increase and moved weight mass toward arithmetic.
    # Running "app" beside "full" separates corpus size from corpus composition, which the
    # first experiment changed together and therefore could not tell apart.
    if mode == "app":
        extra["plain"] = []
    return extra


def app_turns() -> list[str]:
    """The user turn of every app-shaped row, trailer included where there is one."""
    turns = [f"{q}\n\n{TRAILER.format(subject=s)}" for q, s in SEARCH_ROWS] + list(KNOWN_ROWS)
    return turns + extended()["app"]


def build(tokenizer) -> list[tuple[str, int]]:
    """Every calibration row as (prompt text, characters of shared head), app rows first."""
    head = app_head(tokenizer)
    rows = [(render_app(tokenizer, turn), len(head)) for turn in app_turns()]
    rows += [(render(tokenizer, prompt), 0) for prompt in list(PROMPTS) + extended()["plain"]]
    return rows


def sequences(generate, tokenizer, log=print) -> list[tuple[list[int], int]]:
    """Each row, continued by the fp32 model, as (token ids, first position to count).

    ``generate(ids, max_new_tokens)`` returns the greedy continuation's ids. Positions before
    ``keep_from`` are in the sequence -- the row would not be the app's distribution without
    them -- but are not accumulated into any Hessian after the first row that carries them.
    """
    head = app_head(tokenizer)
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    rows: list[tuple[list[int], int]] = []
    empty = 0
    seen_head = False
    for index, (text, head_chars) in enumerate(build(tokenizer)):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        limit = MAX_PROMPT_TOKENS + len(head_ids) if head_chars else MAX_PROMPT_TOKENS
        if len(ids) > limit:
            # Not truncated: the chat template puts the assistant's turn opener at the very
            # end, so cutting the tail would take it off and the teacher would continue the
            # user's sentence instead of answering. Every committed row is far inside this,
            # so reaching here means the corpus or the template changed and wants looking at.
            raise ValueError(
                f"calibration row {index} renders to {len(ids)} tokens, over the {limit} limit; "
                "shorten the question rather than letting it be cut, which would remove the "
                "generation prompt the reply depends on"
            )
        keep_from = 0
        if head_chars:
            if ids[: len(head_ids)] != head_ids:
                raise ValueError(
                    f"calibration row {index} does not open with the declared head. The head is "
                    "tokenised on its own and the row as a whole, so a tokenizer that merges the "
                    "last head token with the first question token lands here; give this family a "
                    "head that ends on a boundary the template already breaks at"
                )
            keep_from = len(head_ids) if seen_head else 0
            seen_head = True
        reply = generate(ids, APP_REPLY_TOKENS if head_chars else MAX_REPLY_TOKENS)
        empty += not len(reply)
        rows.append((list(ids) + list(reply), keep_from))
        log(
            f"calibration row {index + 1}/{len(SEARCH_ROWS) + len(KNOWN_ROWS) + len(PROMPTS)}: "
            f"{len(ids)} prompt + {len(reply)} reply tokens, counting from {keep_from}"
        )
    if empty > len(rows) // 2:
        # The replies are most of what the solve protects. A teacher that emits EOS at once
        # leaves a corpus of prompts, which is the thing this module exists not to be.
        raise ValueError(f"{empty} of {len(rows)} calibration rows came back with no reply at all")
    if empty:
        log(f"==> warning: {empty} rows generated no reply")
    counted = sum(len(ids) - keep for ids, keep in rows)
    log(f"==> {len(rows)} calibration rows, {counted} counted positions")
    return rows
