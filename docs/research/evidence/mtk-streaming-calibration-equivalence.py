"""Old lfm2.py (Arrow round trip, per-chunk calibration) against the streaming rewrite:
calibrate the same model on the same samples and dump every observer's recorded range.

usage: equiv.py old|new <dataset> <window> <out.pt>
Runs from examples/mediatek. MediaTek's compiler is stubbed: calibration never calls it.
"""
import os
import sys
import time
import types

for name in ("mtk_converter", "mtk_converter.python", "mtk_converter.python.converters", "mtk_neuron"):
    sys.modules[name] = types.ModuleType(name)
pytorch_mod = types.ModuleType("mtk_converter.python.converters.pytorch")
pytorch_mod.importer_v2 = types.ModuleType("importer_v2")
sys.modules["mtk_converter.python.converters.pytorch"] = pytorch_mod

mode, dataset, window, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
EXAMPLES = os.environ["EXAMPLES"]
WEIGHTS = os.environ["WEIGHTS"]
os.chdir(EXAMPLES)
sys.path.insert(0, EXAMPLES)
sys.path.insert(0, os.path.join(EXAMPLES, "model_export_scripts"))

import torch  # noqa: E402

torch.manual_seed(0)
import lfm2 as new  # noqa: E402
import lfm2_old as old  # noqa: E402
from aot_utils.llm_utils.preformatter import Preformatter  # noqa: E402
from aot_utils.llm_utils.utils import (  # noqa: E402
    get_embedding_layer,
    get_export_shapes,
    get_master_rot_emb,
    load_checkpoints,
    resolve_model_classes,
)
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e  # noqa: E402
from executorch.backends.mediatek import NeuropilotQuantizer, Precision  # noqa: E402

config, weight_dir, tokenizer_class, chunk_class = resolve_model_classes(os.path.join(WEIGHTS, "config.json"))
try:
    tokenizer = tokenizer_class.from_pretrained(weight_dir)
except Exception:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(weight_dir, "tokenizer.json"),
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
if getattr(tokenizer, "eos_token_id", None) is None:
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
master_rot_emb = get_master_rot_emb(config, dtype=torch.float32)


def observers(graph):
    return {
        name: (m.min_val.detach().clone(), m.max_val.detach().clone())
        for name, m in graph.named_modules()
        if hasattr(m, "min_val") and hasattr(m, "max_val")
    }


started = time.time()
stats = {}
if mode == "old":
    from datasets import load_dataset

    cal = load_dataset("text", data_files=dataset, split="train")
    cal = cal.map(old.apply_preformatter, fn_kwargs={"preformatter": preformatter})
    cal = cal.map(old.tokenize_dataset, fn_kwargs={"tokenizer": tokenizer})
    cal = cal.map(
        old.prepare_model_inputs,
        fn_kwargs={
            "models": models,
            "embedding_layer": embedding_layer,
            "master_rot_emb": master_rot_emb,
            "num_blocks_per_chunk": nbpc,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": int(config.head_dim),
            "max_cache_size": max_cache_size,
            "eos_token_id_tensor": torch.tensor(tokenizer.eos_token_id),
            "response_cap": 9,
        },
        writer_batch_size=1,
        load_from_cache_file=False,
    )
    cal = cal.with_format("numpy")
    for k, model in enumerate(models):
        example_inputs, dynamic_shapes = model.get_example_inputs(max_num_token, max_cache_size, True)
        pre = torch.export.export(model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True).module()
        quantizer = NeuropilotQuantizer()
        quantizer.setup_precision(Precision.A16W4)
        graph = prepare_pt2e(pre, quantizer)
        old.calibrate_model(graph, cal, str(k))
        stats[k] = observers(graph)
else:
    samples = new.load_samples(dataset, tokenizer, preformatter)
    prepared = [new.prepare_chunk(m, "A16W4", max_num_token, max_cache_size) for m in models]
    graphs = [g for g, _ in prepared]
    new.calibrate_streaming(
        samples, models, graphs, embedding_layer, master_rot_emb, nbpc, config.num_key_value_heads,
        int(config.head_dim), max_cache_size, max_num_token, torch.tensor(tokenizer.eos_token_id), 9,
    )
    for k, g in enumerate(graphs):
        stats[k] = observers(g)
torch.save(stats, out)
print(f"{mode}: {sum(len(v) for v in stats.values())} observers in {time.time() - started:.0f} s")
