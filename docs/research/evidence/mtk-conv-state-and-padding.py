"""fp32 replica of MediaTek's llama runner driving the LFM2 export graph, scored against the
Hugging Face original, to see what the runner's padding does to a hybrid model without any
quantization in the way.

It reproduces LlamaRuntime::Run and LlamaModelChunk exactly where they touch the numbers: the
first feed of a fresh cache is left-padded with token 0, later feeds are right-padded and the
caches rolled back, masks are 0 / -100 (MaskBuilder, fp32), padded rows fully masked, and
rotary rows for pads left at zero.

usage: sim.py <layout: old|new> <mode> <window> <prompt name> [decode steps]
  mode "exact": no padding at all (each feed runs at its own length), the harness check
  mode "runner": the shell runner's feed (remainder first, then full 128 batches)
  mode "pieces": the app's feed, the prompt in two pieces, so the second is right-padded
"""
import json
import os
import sys
import types

for name in ("mtk_converter", "mtk_converter.python", "mtk_converter.python.converters", "mtk_neuron"):
    sys.modules[name] = types.ModuleType(name)
pytorch_mod = types.ModuleType("mtk_converter.python.converters.pytorch")
pytorch_mod.importer_v2 = types.ModuleType("importer_v2")
sys.modules["mtk_converter.python.converters.pytorch"] = pytorch_mod

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.environ.get("EXAMPLES", os.path.join(HERE, "executorch", "examples", "mediatek"))
WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b"
)
layout, mode, window, prompt_name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
decode_steps = int(sys.argv[5]) if len(sys.argv) > 5 else 32
os.chdir(EXAMPLES)
sys.path.insert(0, EXAMPLES)

import torch  # noqa: E402
from aot_utils.llm_utils.utils import (  # noqa: E402
    get_embedding_layer,
    get_master_rot_emb,
    load_checkpoints,
    resolve_model_classes,
)
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast  # noqa: E402

torch.manual_seed(0)
BATCH, MASK_FALSE, PAD_TOKEN, NUM_CHUNKS = 128, -100.0, 0, 4

config, weight_dir, _, chunk_class = resolve_model_classes(os.path.join(WEIGHTS, "config.json"))
nbpc = [(config.num_hidden_layers // NUM_CHUNKS) + (i < (config.num_hidden_layers % NUM_CHUNKS)) for i in range(NUM_CHUNKS)]
state_dict = load_checkpoints(weight_dir)
embedding = get_embedding_layer(config, weight_dir, state_dict)
chunks = []
for i, n in enumerate(nbpc):
    chunk = chunk_class(config, n, chunk_idx=i, dtype=torch.float32, include_tail=(i == NUM_CHUNKS - 1), jit_trace=True)
    chunks.append(chunk.load_weights(state_dict, sum(nbpc[:i])).eval())
del state_dict
rot = get_master_rot_emb(config, dtype=torch.float32)  # (1, 2, max_len, head_dim)
head_dim, kv_heads = int(config.head_dim), config.num_key_value_heads
layer_types = config.layer_types

# PRECISION=A16W8 (or A16W4) swaps each fp32 chunk for its quantize-dequantize graph, calibrated
# the way the export calibrates it, on MediaTek's alpaca prompts (or DATASET).
if os.environ.get("PRECISION"):
    sys.path.insert(0, os.path.join(EXAMPLES, "model_export_scripts"))
    import lfm2 as export_script  # noqa: E402
    from aot_utils.llm_utils.preformatter import Preformatter  # noqa: E402
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e  # noqa: E402

    cal_tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(WEIGHTS, "tokenizer.json"))
    dataset = os.environ.get("DATASET", "aot_utils/llm_utils/prompts/alpaca.txt")
    pre = None if dataset.endswith(".jsonl") else Preformatter("aot_utils/llm_utils/preformatter_templates/qwen3.json")
    samples = export_script.load_samples(dataset, cal_tok, pre)
    prepared = [export_script.prepare_chunk(c, os.environ["PRECISION"], BATCH, window)[0] for c in chunks]
    eos = torch.tensor(config.eos_token_id)
    if layout == "old":
        export_script.calibrate_streaming(samples, chunks, prepared, embedding, rot, nbpc, kv_heads, head_dim,
                                          window, BATCH, eos, 9)
    else:
        export_script.calibrate_streaming(samples, chunks, prepared, embedding, rot, window, BATCH, eos, 9,
                                          config.conv_L_cache - 1)
    chunks = [convert_pt2e(g, fold_quantize=False) for g in prepared]
    del prepared


def chunk_layers(i):
    start = sum(nbpc[:i])
    return layer_types[start : start + nbpc[i]]


def fresh_cache():
    """Per chunk, the state tensors in the order the chunk's forward takes them."""
    caches = []
    for i in range(NUM_CHUNKS):
        if layout == "old":
            caches.append([torch.zeros(1, kv_heads, window, head_dim) for _ in range(2 * nbpc[i])])
        else:
            states = []
            for t in chunk_layers(i):
                if t == "conv":
                    states.append(torch.zeros(1, config.hidden_size, config.conv_L_cache - 1))
                else:
                    states += [torch.zeros(1, kv_heads, window, head_dim), torch.zeros(1, kv_heads, window, head_dim)]
            caches.append(states)
    return caches


def is_attention_cache(t):
    return t.dim() == 4 and t.shape[2] == window


def build_mask(n, seen, left, right):
    """MaskBuilder::buildMask then adjustMaskForPadding, fp32."""
    mask = torch.full((1, 1, n, window + n), MASK_FALSE)
    visible = min(window, seen)
    for r in range(n):
        last = window + r
        first = last - (visible + r + 1) + 1
        mask[0, 0, r, first : last + 1] = 0.0
    if left:
        mask[0, 0, :left, :] = MASK_FALSE
        for r in range(left, n):
            mask[0, 0, r, window : window + min(left, r + 1)] = MASK_FALSE
    if right:
        mask[0, 0, n - right :, :] = MASK_FALSE
    return mask


def build_pos(n, index, left, right):
    """RotaryEmbeddingMasterLut::setEmbed: valid rows from index on, pads left at zero."""
    pos = torch.zeros(1, 2, n, head_dim)
    valid = n - left - right
    pos[:, :, left : left + valid, :] = rot[:, :, index : index + valid, :]
    return pos


class Runner:
    def __init__(self):
        self.cache = fresh_cache()
        self.index = 0

    def run(self, tokens, batch):
        pad = batch - len(tokens)
        left = pad if self.index == 0 else 0
        right = pad - left
        ids = [PAD_TOKEN] * left + list(tokens) + [PAD_TOKEN] * right
        hidden = embedding(torch.tensor([ids], dtype=torch.int32))
        mask = build_mask(batch, self.index, left, right)
        pos = build_pos(batch, self.index, left, right)
        extra = []
        if layout == "new":
            extra = conv_inputs(batch, left, right)
        with torch.no_grad():
            for k, chunk in enumerate(chunks):
                outs = chunk(hidden, mask, pos, *extra, *self.cache[k])
                hidden = outs[0]
                self.cache[k] = [o.clone() for o in outs[1:]]
        if right:
            self.rollback(right, self.index + batch)
        elif left:
            for states in self.cache:
                for t in states:
                    if layout == "old" or is_attention_cache(t):
                        t[:, :, window - batch : window - batch + left, :] = 0
        self.index += len(tokens)
        return hidden[0, batch - 1 - right]

    def rollback(self, count, seen):
        """LlamaModelChunk::RollbackCache: shift right by count, zero what moved out."""
        alive = min(seen, window)
        first = window - alive
        keep = max(alive - count, 0)
        for states in self.cache:
            for t in states:
                if layout == "new" and not is_attention_cache(t):
                    continue
                if keep:
                    t[:, :, first + count : first + count + keep, :] = t[:, :, first : first + keep, :].clone()
                t[:, :, first : first + count, :] = 0


def conv_inputs(batch, left, right):
    """The two inputs the corrected graph takes for its conv layers (see modeling_lfm2)."""
    valid = torch.zeros(1, batch, 1)
    valid[0, left : batch - right, 0] = 1.0
    last = batch - right  # index in the state-padded sequence of the last valid Bx is last + 1
    select = torch.zeros(1, batch + config.conv_L_cache - 1, config.conv_L_cache - 1)
    for j in range(config.conv_L_cache - 1):
        select[0, last + j, j] = 1.0
    return [valid, select]


def feed(runner, tokens):
    """The shell runner's digest_prompt and the app's FeedTokens: remainder first."""
    logits, cursor = None, 0
    while cursor < len(tokens):
        remain = len(tokens) - cursor
        take = remain % BATCH or BATCH
        batch = BATCH if mode != "exact" else take
        logits = runner.run(tokens[cursor : cursor + take], batch)
        cursor += take
    return logits


fidelity = json.load(open(os.path.join(HERE, "phone-rig", "phone", "fidelity.json")))
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(WEIGHTS, "tokenizer.json"))
text = open(os.path.join(HERE, "phone-rig", "phone", "prompts", prompt_name + ".txt"), encoding="utf-8").read()
prompt = [config.bos_token_id] + tok(text, add_special_tokens=False)["input_ids"]
cont = fidelity[prompt_name]["ids"][:decode_steps] if prompt_name in fidelity else []
if len(prompt) + len(cont) >= window:
    sys.exit(f"{len(prompt)} + {len(cont)} tokens do not fit a {window} window")

hf = AutoModelForCausalLM.from_pretrained(WEIGHTS, dtype=torch.float32).eval()
with torch.no_grad():
    ref = hf(torch.tensor([prompt + cont])).logits[0].float()
del hf

runner = Runner()
if mode == "pieces":
    cut = len(prompt) // 2 + 37  # any cut that is not a multiple of 128
    feed(runner, prompt[:cut])
    step_logits = [feed(runner, prompt[cut:])]
else:
    step_logits = [feed(runner, prompt)]
for t in cont[:-1]:
    step_logits.append(runner.run([t], 1))

kls, agree = [], 0
for i, lg in enumerate(step_logits):
    r = torch.log_softmax(ref[len(prompt) - 1 + i], -1)
    q = torch.log_softmax(lg.float(), -1)
    kls.append(float((r.exp() * (r - q)).sum()))
    agree += int(r.argmax() == q.argmax())
print(json.dumps({"layout": layout, "mode": mode, "window": window, "prompt": prompt_name, "prompt_tokens": len(prompt),
                  "steps": len(step_logits), "argmax_agree": agree, "kl_first": kls[0], "kl_mean": sum(kls) / len(kls),
                  "kl_max": max(kls), "kl_steps": [round(k, 6) for k in kls[:8]]}))
