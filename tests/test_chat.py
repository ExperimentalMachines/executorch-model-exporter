import pytest
from jinja2.exceptions import TemplateSyntaxError

from pipeline import chat

QWEN3_LIKE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)
LLAMA_LIKE = (
    "{{ bos_token }}{% for m in messages %}"
    "<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
    "{% endfor %}"
)


def test_instruct_prompt_renders_upstream_template_without_thinking(tmp_path):
    prompt = chat.render(tmp_path, {"chat_template": QWEN3_LIKE}, instruct=True)
    assert prompt.startswith("<|im_start|>user\nWhat is the capital of France?")
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_bos_is_written_as_text(tmp_path):
    prompt = chat.render(tmp_path, {"chat_template": LLAMA_LIKE, "bos_token": "<|begin_of_text|>"}, instruct=True)
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>user")


def test_standalone_template_file_wins(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("X{{ messages[0].content }}", encoding="utf-8")
    assert chat.render(tmp_path, {"chat_template": QWEN3_LIKE}, instruct=True) == "X" + chat.QUESTION


def test_base_models_get_a_completion_prompt(tmp_path):
    assert chat.render(tmp_path, {"chat_template": QWEN3_LIKE}, instruct=False) == chat.COMPLETION
    with_bos = {"bos_token": {"content": "<s>"}, "add_bos_token": True}
    assert chat.render(tmp_path, with_bos, instruct=False) == "<s>" + chat.COMPLETION


def test_bos_is_written_even_when_add_bos_token_is_missing(tmp_path):
    """transformers 5.x does not write add_bos_token, and LFM2.5 without BOS is gibberish.

    An abliterated LFM2.5 re-saved by transformers 5.16.1 failed all five windows of its
    export on 2026-09-19 with ' is is is is is' as its only output. Nothing was wrong with
    the weights: its tokenizer_config had no add_bos_token, so this line built a prompt with
    no BOS. Measured on both checkpoints, with BOS and without: with it both answer
    "Paris. It is the most populous city in France", without it both degenerate.
    """
    no_key = {"bos_token": "<|startoftext|>"}
    assert chat.render(tmp_path, no_key, instruct=False) == "<|startoftext|>" + chat.COMPLETION
    # And a model that genuinely has no BOS still gets a clean prompt.
    assert chat.render(tmp_path, {}, instruct=False) == chat.COMPLETION


def test_a_template_the_sandbox_cannot_compile_still_carries_bos(tmp_path):
    # LFM2.5's real template uses transformers' {% generation %} tag. Without transformers
    # installed the sandbox raises, smoke.run catches it and asks for the completion prompt;
    # that prompt has to carry BOS or the test measures a model with no BOS.
    (tmp_path / "chat_template.jinja").write_text("{% generation %}x{% endgeneration %}", encoding="utf-8")
    config = {"bos_token": "<|startoftext|>"}
    with pytest.raises(TemplateSyntaxError):
        chat.render(tmp_path, config, instruct=True)
    assert chat.render(tmp_path, config, instruct=False).startswith("<|startoftext|>")
