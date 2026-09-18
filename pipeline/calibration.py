"""The text GPTQ takes its Hessians from.

A calibration set decides what the solve optimises for, so it is committed here rather than
fetched: a run must be repeatable, and a corpus that changes underneath the pipeline would
change published weights without changing any code.

Two decisions worth stating, because both were measured rather than assumed.

*Rendered in the model's own chat template.* The exports are chat models and the app that
runs them always sends a template, so calibrating on raw text would take the Hessians from a
token distribution the model never meets in use.

*Continued by the model itself.* Each prompt is followed by the fp32 model's own greedy
reply before the sequence is used. What GPTQ protects is the output the unquantised model
would have produced, and the reply is where most of those positions are; calibrating on
prompts alone leaves the decision to answer, or to call a tool, almost unrepresented.

The prompts are deliberately plain and broad. An earlier calibration for the same family
used one application's prompt shape only, and the model it produced tilted toward that
shape: it scored 51 percent on the questions that shape covered and lost ground elsewhere,
against 49 percent for a wider draw that held up across both. A published export is not
built for one caller, so the set below spans question answering, instruction following,
reasoning, code and tool use rather than any one of them.
"""

from __future__ import annotations

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

MAX_PROMPT_TOKENS = 512
MAX_REPLY_TOKENS = 128


def render(tokenizer, prompt: str) -> str:
    """The prompt in the model's own chat template, ready for the assistant's turn."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def sequences(model, tokenizer, prompts=PROMPTS, reply_tokens: int = MAX_REPLY_TOKENS, log=print) -> list[list[int]]:
    """Each prompt, templated, with the fp32 model's own greedy continuation appended."""
    import torch

    rows: list[list[int]] = []
    for index, prompt in enumerate(prompts):
        text = render(tokenizer, prompt)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:MAX_PROMPT_TOKENS]
        generated = list(ids)
        with torch.no_grad():
            for _ in range(reply_tokens):
                logits = model(torch.tensor([generated], dtype=torch.long))
                token = int(logits[0, -1].argmax()) if logits.dim() == 3 else int(logits[0].argmax())
                generated.append(token)
                if token in (tokenizer.eos_token_id,):
                    break
        rows.append(generated)
        log(f"calibration row {index + 1}/{len(prompts)}: {len(ids)} prompt + {len(generated) - len(ids)} reply tokens")
    return rows
