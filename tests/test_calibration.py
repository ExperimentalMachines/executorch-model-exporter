"""The calibration corpus, and the one thing about it the solver has to honour.

Skipped without torch, like tests/test_solve.py: the dev requirements do not carry it.
"""

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from pipeline import calibration, solve  # noqa: E402


class FakeTokenizer:
    """Character-level ids, so a shared prefix of text is a shared prefix of tokens."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        head = json.dumps(tools) if tools else ""
        body = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return head + body + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def constant_reply(ids, max_new_tokens):
    return [7] * min(max_new_tokens, 4)


def test_every_app_row_opens_with_the_same_head():
    # The solver slices a fixed number of tokens off each app row. If a row did not actually
    # open with the head, that slice would eat the question instead and the solve would be
    # calibrated on the reply alone, silently.
    tok = FakeTokenizer()
    head_ids = tok(calibration.app_head(tok))["input_ids"]
    for text, head_chars in calibration.build(tok):
        if head_chars:
            assert tok(text)["input_ids"][: len(head_ids)] == head_ids


def test_the_head_is_counted_in_one_row_only():
    tok = FakeTokenizer()
    rows = calibration.sequences(constant_reply, tok, log=lambda *_: None)
    head_len = len(tok(calibration.app_head(tok))["input_ids"])
    app = [keep for (ids, keep) in rows[: len(calibration.SEARCH_ROWS) + len(calibration.KNOWN_ROWS)]]
    plain = [keep for (ids, keep) in rows[len(calibration.SEARCH_ROWS) + len(calibration.KNOWN_ROWS) :]]
    assert app[0] == 0, "one row has to carry the head, or it is never in a Hessian at all"
    assert set(app[1:]) == {head_len}
    assert set(plain) == {0}, "the plain rows share no head, so every position counts"


def test_the_corpus_holds_both_shapes():
    # A corpus of app rows alone was measured to tilt toward calling (v1, see the module
    # docstring); one with none of them is what this repository shipped and lost a call on.
    tok = FakeTokenizer()
    rows = calibration.build(tok)
    assert sum(1 for _, head in rows if head) == len(calibration.SEARCH_ROWS) + len(calibration.KNOWN_ROWS)
    assert sum(1 for _, head in rows if not head) == len(calibration.PROMPTS)
    assert all("web_search" in text and calibration.APP_SYSTEM in text for text, head in rows if head)


def test_no_calibration_question_is_asked_twice():
    questions = [q for q, _ in calibration.SEARCH_ROWS] + list(calibration.KNOWN_ROWS) + list(calibration.PROMPTS)
    assert len(questions) == len(set(questions))


class Layer(torch.nn.Module):
    """Position-independent on purpose: see the test below."""

    def __init__(self, width):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)

    def forward(self, x, *args, **kwargs):
        return self.linear(x)


class Tiny(torch.nn.Module):
    def __init__(self, vocab=64, width=128):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(vocab, width)
        self.layers = torch.nn.ModuleList([Layer(width)])
        self.norm = torch.nn.Identity()
        self.output = torch.nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        return self.output(self.norm(self.layers[0](self.tok_embeddings(ids))))


HEAD = list(range(40, 56))
BODIES = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]


def tiny_with_a_loud_head():
    """A model whose head tokens excite one corner of the input space hard.

    The point of both tests below is what the Hessian holds, so the head has to be able to
    move it. Head tokens that looked like body tokens would make either answer come out the
    same and prove nothing.
    """
    torch.manual_seed(0)
    model = Tiny()
    with torch.no_grad():
        model.tok_embeddings.weight[HEAD] = 0.0
        model.tok_embeddings.weight[HEAD, :16] = 12.0
    return model


def test_skipped_positions_are_really_out_of_the_hessian():
    # This model has no attention, so a position's activations depend on its own token only.
    # Under that condition "keep the head in the row but count it once" and "only one row
    # ever had a head" are the same Hessian, so the codes must come out identical. A solver
    # that ignored keep_from would be counting the head 3 times here and would not.
    model = tiny_with_a_loud_head()
    counted_once = [(HEAD + BODIES[0], 0)] + [(HEAD + b, len(HEAD)) for b in BODIES[1:]]
    without_head = [(HEAD + BODIES[0], 0)] + [(b, 0) for b in BODIES[1:]]

    a = solve.solve_model(copy.deepcopy(model), counted_once, log=lambda *_: None)
    b = solve.solve_model(copy.deepcopy(model), without_head, log=lambda *_: None)

    assert set(a) == {"layers.0.linear", "output"}
    for name in a:
        assert torch.equal(a[name]["qdata"], b[name]["qdata"]), name
        assert torch.equal(a[name]["scale"], b[name]["scale"]), name


def test_counting_the_head_every_time_would_have_changed_the_answer():
    # The guard for the test above: if the head made no difference either way, that test
    # would pass against a solver that had simply ignored it.
    model = tiny_with_a_loud_head()
    counted_once = [(HEAD + BODIES[0], 0)] + [(HEAD + b, len(HEAD)) for b in BODIES[1:]]
    counted_always = [(HEAD + b, 0) for b in BODIES]

    a = solve.solve_model(copy.deepcopy(model), counted_once, log=lambda *_: None)
    b = solve.solve_model(copy.deepcopy(model), counted_always, log=lambda *_: None)
    assert any(not torch.equal(a[n]["qdata"], b[n]["qdata"]) for n in a)


def test_a_row_too_long_for_the_limit_is_refused_not_cut():
    # Cutting the tail would take off the chat template's generation prompt, and the teacher
    # would continue the user's sentence instead of answering it. That reply would then be
    # most of what the solve protects.
    tok = FakeTokenizer()
    long_question = ("x" * 4000, "subject")
    original = calibration.SEARCH_ROWS
    calibration.SEARCH_ROWS = (long_question,)
    try:
        with pytest.raises(ValueError, match="over the"):
            calibration.sequences(constant_reply, tok, log=lambda *_: None)
    finally:
        calibration.SEARCH_ROWS = original


def test_a_teacher_that_says_nothing_is_refused():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match="no reply at all"):
        calibration.sequences(lambda ids, n: [], tok, log=lambda *_: None)
