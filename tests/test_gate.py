"""The gate that decides whether an export is allowed to publish.

No torch needed: check() is arithmetic over two dicts, and the numbers below are the ones
four real .pte files produced on these prompts (see the module docstring).
"""

from pipeline import calibration, gate

# What the four measured files scored on the twelve calling rows, and fp32's own mean.
FP32 = 0.8212
MEASURED = {
    "round to nearest, the file that lost its tool calling": (0.049, 0),
    "GPTQ calibrated without a tool schema": (0.7297, 12),
    "GPTQ v2, the published file": (0.701, 12),
    "GPTQ on an abliterated checkpoint": (0.945, 12),
}


def reference(p=FP32, quiet_p=0.99):
    rows = {q: {"token": 1, "p": p, "text": "<tool>"} for q, _ in gate.SEARCH}
    rows.update({q: {"token": 2, "p": quiet_p, "text": "The"} for q in gate.QUIET})
    return {"rows": rows, "confident": len(rows)}


def measured(mean, agreed, quiet_followed=None):
    rows = {}
    for index, (q, _) in enumerate(gate.SEARCH):
        rows[q] = {"p": mean, "top": 1 if index < agreed else 999}
    followed = len(gate.QUIET) if quiet_followed is None else quiet_followed
    for index, q in enumerate(gate.QUIET):
        rows[q] = {"p": 0.9, "top": 2 if index < followed else 999}
    return rows


def test_the_broken_file_is_refused_and_the_working_ones_are_not():
    ref = reference()
    verdicts = {name: gate.check(ref, measured(mean, agreed)) for name, (mean, agreed) in MEASURED.items()}
    refused = [name for name, v in verdicts.items() if not v["passed"]]
    assert refused == ["round to nearest, the file that lost its tool calling"], verdicts


def test_an_export_above_fp32_is_not_refused_for_it():
    # An abliterated checkpoint calls more readily than the model it came from. That is the
    # abliteration, not a defect, and a symmetric band would have blocked its publish.
    verdict = gate.check(reference(), measured(0.945, len(gate.SEARCH)))
    assert verdict["passed"]
    assert verdict["shortfall"] < 0


def test_losing_calls_fails_even_when_the_mean_looks_healthy():
    # The failure mode this exists for: a file that is confident on the rows it still calls
    # on, and has silently stopped calling on the others.
    verdict = gate.check(reference(), measured(0.82, agreed=10))
    assert not verdict["passed"]
    assert any("kept fp32's choice on 10" in p for p in verdict["problems"])


def test_reaching_for_a_tool_on_quiet_rows_fails():
    verdict = gate.check(reference(), measured(0.80, len(gate.SEARCH), quiet_followed=1))
    assert not verdict["passed"]
    assert any("quiet rows" in p for p in verdict["problems"])


def test_a_model_fp32_never_calls_on_is_not_graded_as_a_pass():
    # If fp32 is undecided everywhere, there is nothing to compare against and the gate has
    # to say so. Passing quietly here would wave through every export of that family.
    ref = reference(p=0.1)
    verdict = gate.check(ref, measured(0.1, 0))
    assert not verdict["passed"]
    assert verdict["graded"] == 0


def test_the_gate_never_grades_a_question_the_solve_calibrated_on():
    # A gate drawn from the calibration set would be marking a solve's own homework.
    calibrated = {q for q, _ in calibration.SEARCH_ROWS} | set(calibration.KNOWN_ROWS) | set(calibration.PROMPTS)
    graded = {q for q, _ in gate.SEARCH} | set(gate.QUIET)
    assert not (calibrated & graded)


def test_a_model_that_never_calls_is_marked_as_unmeasured():
    """LFM2.5-2.6B answers "The" at probability 1.0 to every row, trailered or not.

    Its export still has to track its fp32 model and does, so this is a pass. But the gate
    checked nothing about tool calling, and a verdict that did not say so would be read as
    if it had.
    """
    rows = {q: {"token": 2, "p": 1.0, "text": "The"} for q, _ in gate.SEARCH}
    rows.update({q: {"token": 2, "p": 1.0, "text": "The"} for q in gate.QUIET})
    ref = {"rows": rows, "confident": len(rows)}
    measured = {q: {"p": 1.0, "top": 2} for q in rows}
    verdict = gate.check(ref, measured)
    assert verdict["passed"]
    assert verdict["measures_tool_calling"] is False


def test_a_model_that_does_call_is_marked_as_measured():
    verdict = gate.check(reference(), measured(0.80, len(gate.SEARCH)))
    assert verdict["passed"]
    assert verdict["measures_tool_calling"] is True
