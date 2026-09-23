"""Controlled A/B of MediaTek calibration sets: same model, same A16W4 quantizer, only the
calibration data differs. Calibrates with lfm2.py's streaming pass, converts (convert_pt2e, the
graph MediaTek then lowers), and runs held-out text teacher-forced through the fp32 chunks and
the quantized chunks, each on its own cache as the device would. Records per position the fp32
NLL, the quantized NLL and KL(fp32 || quantized).

usage: quant_eval.py <dataset: alpaca.txt path | corpus.jsonl> <window> <eval_tokens> <out.json>
"""
import json
import os
import re
import sys
import time
import types

for name in ("mtk_converter", "mtk_converter.python", "mtk_converter.python.converters", "mtk_neuron"):
    sys.modules[name] = types.ModuleType(name)
pytorch_mod = types.ModuleType("mtk_converter.python.converters.pytorch")
pytorch_mod.importer_v2 = types.ModuleType("importer_v2")
sys.modules["mtk_converter.python.converters.pytorch"] = pytorch_mod

dataset, window, eval_tokens, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
EXAMPLES, WEIGHTS, BOOK = os.environ["EXAMPLES"], os.environ["WEIGHTS"], os.environ["BOOK"]
os.chdir(EXAMPLES)
sys.path.insert(0, EXAMPLES)
sys.path.insert(0, os.path.join(EXAMPLES, "model_export_scripts"))

import torch  # noqa: E402

torch.manual_seed(0)
import lfm2 as new  # noqa: E402
from aot_utils.llm_utils.preformatter import Preformatter  # noqa: E402
from aot_utils.llm_utils.utils import (  # noqa: E402
    generate_mask,
    get_embedding_layer,
    get_export_shapes,
    get_master_rot_emb,
    load_checkpoints,
    resolve_model_classes,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e  # noqa: E402

config, weight_dir, tokenizer_class, chunk_class = resolve_model_classes(os.path.join(WEIGHTS, "config.json"))
from transformers import PreTrainedTokenizerFast  # noqa: E402

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(weight_dir, "tokenizer.json"),
    bos_token_id=config.bos_token_id,
    eos_token_id=config.eos_token_id,
    pad_token_id=config.pad_token_id,
)
tokenizer.eos_token_id = config.eos_token_id
preformatter = None if dataset.endswith(".jsonl") else Preformatter("aot_utils/llm_utils/preformatter_templates/qwen3.json")
num_chunks = 4
nbpc = [(config.num_hidden_layers // num_chunks) + (i < (config.num_hidden_layers % num_chunks)) for i in range(num_chunks)]
state_dict = load_checkpoints(weight_dir)
export_shapes, max_num_token, max_cache_size = get_export_shapes([f"128t{window}c", f"1t{window}c"])
embedding_layer = get_embedding_layer(config, weight_dir, state_dict)
models = []
for i, n in enumerate(nbpc):
    chunk = chunk_class(config, n, chunk_idx=i, dtype=torch.float32, include_tail=(i == num_chunks - 1), jit_trace=True)
    models.append(chunk.load_weights(state_dict, sum(nbpc[:i])))
del state_dict
master_rot_emb = get_master_rot_emb(config, dtype=torch.float32)

started = time.time()
samples = new.load_samples(dataset, tokenizer, preformatter)
prepared = [new.prepare_chunk(m, os.environ.get("PRECISION", "A16W4"), max_num_token, max_cache_size)[0] for m in models]
new.calibrate_streaming(
    samples, models, prepared, embedding_layer, master_rot_emb, nbpc, config.num_key_value_heads,
    int(config.head_dim), max_cache_size, max_num_token, torch.tensor(tokenizer.eos_token_id), 9,
)
quant = [convert_pt2e(g, fold_quantize=False) for g in prepared]
del prepared
calibrated = time.time() - started

# Held-out text, not in any calibration set: a chat-template user turn holding a book passage,
# so positions are what a long prompt occupies on the phone.
text = open(BOOK, encoding="utf-8").read().replace("\r\n", "\n")
text = re.sub(r"\n{3,}", "\n\n", text)
body = tokenizer(text[len(text) // 3:], add_special_tokens=False)["input_ids"]
head = tokenizer("<|im_start|>user\nHere is a passage from a book.\n\n", add_special_tokens=False)["input_ids"]
ids = ([config.bos_token_id] + head + body)[:eval_tokens]


def reset():
    return new.reset_cache(num_chunks, config.num_key_value_heads, nbpc, int(config.head_dim), max_cache_size)


def step(graphs, hidden, cache, mask, pos_emb):
    with torch.no_grad():
        for k, g in enumerate(graphs):
            outs = g(hidden, mask, pos_emb, *torch.split(cache[k], 1, dim=0))
            hidden = outs[0]
            cache[k] = torch.cat(outs[1 : 1 + 2 * nbpc[k]], dim=0).clone()
    return hidden, cache


ref_cache, q_cache = reset(), reset()
nll_ref, nll_q, kl = [], [], []
pos = 0
for start in range(0, len(ids) - 1, max_num_token):
    block = ids[start : start + max_num_token]
    hidden = embedding_layer(torch.tensor([block], dtype=torch.int32))
    n = hidden.shape[1]
    mask = generate_mask(max_cache_size, pos, n, n)
    pos_emb = master_rot_emb[:, :, pos : pos + n, :]
    logits_ref, ref_cache = step(models, hidden, ref_cache, mask, pos_emb)
    logits_q, q_cache = step(quant, hidden, q_cache, mask, pos_emb)
    lp_ref = torch.log_softmax(logits_ref[0].float(), -1)
    lp_q = torch.log_softmax(logits_q[0].float(), -1)
    targets = ids[start + 1 : start + n + 1]
    for t, target in enumerate(targets):
        nll_ref.append(float(-lp_ref[t, target]))
        nll_q.append(float(-lp_q[t, target]))
        kl.append(float((lp_ref[t].exp() * (lp_ref[t] - lp_q[t])).sum()))
    pos += n
json.dump(
    {"dataset": os.path.basename(dataset), "window": window, "tokens": len(nll_ref), "calibration_s": calibrated,
     "nll_ref": nll_ref, "nll_q": nll_q, "kl": kl},
    open(out, "w"),
)
print(f"{os.path.basename(dataset)}@{window}: {len(nll_ref)} positions, calibrated in {calibrated:.0f} s, "
      f"total {time.time() - started:.0f} s")
