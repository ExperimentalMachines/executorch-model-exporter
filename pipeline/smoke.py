"""Load an exported .pte the way the phone does and check it produces sane text.

``TextLLMRunner`` is the C++ runner the app's ``LlmModule`` wraps, shipped in the Linux
executorch wheel. The metadata methods are read back with ``executorch.runtime`` so the
published config.json reports what the file says, not what the export was asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import chat

_METADATA_PREFIXES = ("get_", "use_", "enable_")
# Below this size a model may not know the answer; only degenerate output fails it.
ANSWER_REQUIRED_FROM_PARAMS = 500_000_000


def _plain(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def read_metadata(pte: Path) -> dict:
    from executorch.runtime import Runtime

    program = Runtime.get().load_program(pte)
    metadata = {}
    for name in sorted(program.method_names):
        if not name.startswith(_METADATA_PREFIXES):
            continue
        outputs = [_plain(v) for v in program.load_method(name).execute([])]
        if name == "get_eos_ids":
            flat = []
            for value in outputs:
                flat.extend(value if isinstance(value, list) else [value])
            metadata[name] = [int(v) for v in flat]
        else:
            value = outputs[0] if len(outputs) == 1 else outputs
            metadata[name] = value
    metadata["methods"] = sorted(program.method_names)
    return metadata


def delegates(pte: Path) -> dict[str, dict]:
    """Per method: the backends it delegates to and how many delegate calls it makes.

    Read from the program's flatbuffer (the schema ExecuTorch serialises), not by grepping
    bytes: the backend id string also appears in a program whose graph never calls it.
    """
    from executorch.exir._serialize._program import deserialize_pte_binary
    from executorch.exir.schema import DelegateCall

    program = deserialize_pte_binary(pte.read_bytes()).program
    result = {}
    for plan in program.execution_plan:
        calls = sum(
            1
            for chain in plan.chains
            for instruction in chain.instructions
            if isinstance(instruction.instr_args, DelegateCall)
        )
        result[plan.name] = {"backends": sorted({d.id for d in plan.delegates}), "delegate_calls": calls}
    return result


def structural_check(pte: Path, wanted: tuple[str, ...], backend_id: str) -> dict:
    """The program loads, has the ``wanted`` methods, and each of them runs on ``backend_id``.

    For files no host runtime can execute: NPU context binaries, and Vulkan on a runner
    without a GPU.
    """
    problems = []
    try:
        metadata = read_metadata(pte)
        graphs = delegates(pte)
    except Exception as error:  # a program that does not even parse
        return {"kind": "structural", "passed": False, "problems": [f"program did not load: {error}"]}
    methods = metadata.get("methods", [])
    for name in wanted:
        graph = graphs.get(name)
        if graph is None:
            problems.append(f"missing decoder method {name!r} (has {methods})")
        elif backend_id not in graph["backends"] or graph["delegate_calls"] == 0:
            problems.append(f"{name} does not run on {backend_id}: {graph}")
    return {
        "kind": "structural",
        "passed": not problems,
        "problems": problems,
        "methods": methods,
        "delegates": {name: graphs[name] for name in wanted if name in graphs},
        "metadata": metadata,
    }


def generate(pte: Path, tokenizer: Path, prompt: str, max_new_tokens: int) -> tuple[list[str], dict]:
    # The wheel's runner links only portable and XNNPACK kernels. The exported graph also
    # calls llama::custom_sdpa / update_cache and quantized_decomposed::embedding_byte,
    # whose kernels register into portable_lib's operator registry when these libraries
    # load, in this order (examples/models/llama/runner/native.py). The app's AAR links
    # them statically.
    # isort: off
    from executorch.extension.pybindings import portable_lib  # noqa: F401
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    from executorch.kernels import quantized  # noqa: F401
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner
    # isort: on

    runner = TextLLMRunner(str(pte), str(tokenizer))
    pieces: list[str] = []
    stats: dict = {}
    config = GenerationConfig(echo=False, max_new_tokens=max_new_tokens, temperature=0.0, num_bos=0, num_eos=0)
    runner.generate(
        prompt,
        config,
        token_callback=pieces.append,
        stats_callback=lambda s: stats.update(json.loads(s.to_json_string())),
    )
    return pieces, stats


def degenerate(pieces: list[str]) -> bool:
    """Eight or more tokens that are all the same piece."""
    return len(pieces) >= 8 and len(set(pieces)) == 1


def run(
    pte: Path,
    tokenizer: Path,
    model_dir: Path,
    tokenizer_config: dict,
    instruct: bool,
    total_params: int,
    max_new_tokens: int = 32,
) -> dict:
    template_error = None
    try:
        prompt = chat.render(model_dir, tokenizer_config, instruct)
    except Exception as error:  # a template feature the sandboxed renderer lacks, not a bad .pte
        template_error = f"{type(error).__name__}: {error}"
        prompt = chat.render(model_dir, tokenizer_config, instruct=False)
    problems = []
    try:
        pieces, stats = generate(pte, tokenizer, prompt, max_new_tokens)
    except Exception as error:
        # The runner is C++ behind pybind11, which maps its exceptions onto whichever Python
        # type matches: std::runtime_error to RuntimeError, but std::out_of_range to
        # IndexError and std::invalid_argument to ValueError. Catching one of them let an
        # `IndexError: stoi` from the tokenizer kill the export job outright instead of
        # being recorded as a failed smoke test (2026-09-19, SmolLM2-135M on an ARM runner).
        # A smoke test that cannot run is a result about the file, not a crash of the tool.
        pieces, stats = [], {}
        problems.append(f"runner error: {type(error).__name__}: {error}")
    text = "".join(pieces)
    answered = chat.EXPECTED in text.lower()
    required = total_params >= ANSWER_REQUIRED_FROM_PARAMS
    if not pieces and not problems:
        problems.append("generated no tokens")
    if degenerate(pieces):
        problems.append(f"degenerate output: {pieces[0]!r} repeated")
    if required and not answered:
        problems.append(f"expected {chat.EXPECTED!r} in the reply")
    return {
        "prompt": prompt,
        "template_error": template_error,
        "reply": text,
        "tokens": len(pieces),
        "answered": answered,
        "answer_required": required,
        "passed": not problems,
        "problems": problems,
        "stats": stats,
    }
