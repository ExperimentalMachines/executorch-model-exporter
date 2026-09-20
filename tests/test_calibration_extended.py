"""The experiment's 377-row calibration corpus: off by default, and clean of graded questions.

No torch here on purpose. These are the two properties that must hold on any machine,
and they are worth checking in an environment that cannot run a solve.
"""

import json
import re
from pathlib import Path

import pytest

from pipeline import calibration

EXTENDED = Path(__file__).resolve().parent.parent / "config" / "calibration-extended.json"
GRADED = Path("/Users/alpha/mobile-inference/tools/eval/results/elite-2026-09-20/inputs/decisions.json")


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def test_the_extended_corpus_is_off_unless_the_experiment_asks_for_it(monkeypatch):
    """The published recipe is 104 rows. A calibration change silently alters every weight in
    every published file, so the 377-row corpus must never reach a solve by accident."""
    monkeypatch.delenv("OW_CALIB_EXTENDED", raising=False)
    assert calibration.extended() == {"app": [], "plain": []}
    assert len(calibration.app_turns()) == len(calibration.SEARCH_ROWS) + len(calibration.KNOWN_ROWS)


def test_the_extended_corpus_loads_when_asked(monkeypatch):
    monkeypatch.setenv("OW_CALIB_EXTENDED", "1")
    extra = calibration.extended()
    assert len(extra["app"]) + len(extra["plain"]) == 273
    expected = len(calibration.SEARCH_ROWS) + len(calibration.KNOWN_ROWS) + len(extra["app"])
    assert len(calibration.app_turns()) == expected


def test_no_extended_question_appears_in_the_graded_decision_set():
    """These rows come from the lab corpus, which drew on a graded set: three of its questions
    were in the decision set and were removed when this file was built. Pin that, because a
    calibration row that is also a test question turns the measurement into nonsense."""
    if not GRADED.exists():
        pytest.skip("the graded decision set is not on this machine")
    data = json.loads(GRADED.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else (data.get("rows") or data.get("items") or [])
    banned = {normalise(i.get("question") or "") for i in items if isinstance(i, dict)}
    extra = json.loads(EXTENDED.read_text(encoding="utf-8"))
    for row in list(extra["app"]) + list(extra["plain"]):
        assert normalise(row) not in banned, f"calibration row is in the graded set: {row[:60]}"


def test_app_mode_adds_the_app_rows_and_none_of_the_word_problems(monkeypatch):
    """Arm B of the experiment. The 120 plain rows are GSM8K-style word problems; leaving
    them out changes corpus size without changing its shape, which is what tells corpus size
    apart from corpus composition."""
    monkeypatch.setenv("OW_CALIB_EXTENDED", "app")
    extra = calibration.extended()
    assert len(extra["app"]) == 153
    assert extra["plain"] == []


def test_an_unrecognised_mode_is_off_rather_than_an_error(monkeypatch):
    """A typo in the workflow must not silently calibrate on something unintended."""
    for value in ("", "0", "true", "yes", "FULL"):
        monkeypatch.setenv("OW_CALIB_EXTENDED", value)
        assert calibration.extended() == {"app": [], "plain": []}, value
