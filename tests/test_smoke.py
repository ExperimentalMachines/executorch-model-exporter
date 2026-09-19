import pytest

from pipeline import smoke


def test_the_token_budget_leaves_room_for_a_model_that_reasons():
    """LFM2.5-2.6B spends about 30 tokens reasoning before it writes "Paris", and stops at
    110 (measured on the fp32 model, 2026-09-19).

    The old budget of 32 cut the reply off mid-sentence, so the smoke test failed every
    window of a working export (run 35408948368). A model that answers immediately is
    unaffected; a model that thinks first now has room to finish.
    """
    assert smoke.MAX_NEW_TOKENS >= 110


@pytest.mark.parametrize(
    ("pieces", "expected"),
    [
        ([" is"] * 8, True),
        ([" is"] * 7, False),
        (["Paris", "<|im_end|>"], False),
        ([], False),
    ],
)
def test_degenerate_still_catches_a_repeating_file(pieces, expected):
    # The larger budget must not weaken the one check that catches a destroyed export.
    assert smoke.degenerate(pieces) is expected
