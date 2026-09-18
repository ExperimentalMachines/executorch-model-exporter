"""GPTQ onto the grid XNNPACK runs: symmetric int4, groups of 32, positive bf16 scales.

ExecuTorch has no calibrated int4 path to this delegate (pytorch/executorch #3632 and
#9846, both open): ``8da4w-gptq`` is accepted and refused, and torchao's own GPTQ writes
weight-only formats the delegate cannot lower. The delegate does not care who chose the
codes, though, so they are solved here on the eager module ``export_llm`` traces and written
into torchao's tensors before lowering. Same file format, same kernels, same size; different
rounding.

Why it earns a stage. Round to nearest is what every published export used, and on the
LFM2.5 family it costs the model its tool calling: on 141 held-out questions the rounded
1.2B export searched when needed on 10 percent of the questions that needed it and spoke of
search results it had never fetched in a quarter of its replies, where the solved export
reads 49 and 1 percent (finding 29). Attention-only families lose much less, so this is an
improvement for them rather than a repair.

The method, per layer in order, with everything upstream already quantised so each layer is
solved on the inputs it will really see: the Hessian of every int4 linear is accumulated
from its input after the per-token int8 rounding the delegate applies, the weight is solved
column by column with the error pushed onto the columns not yet rounded (Frantar et al.,
arXiv 2210.17323), and each group's scale is the clip that minimises that group's squared
error at the moment the solver reaches it. No activation reordering: XNNPACK's groups are
contiguous.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# The clip grid each group's scale is chosen from, widest first.
CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
# int4, symmetric: the largest magnitude a code can carry is 7.5 half-steps from zero.
INT4_HALF_RANGE = 7.5
GROUP = 32


def bf16_storage(scale):
    """What the delegate keeps of a blockwise scale.

    ExecuTorch 1.4 writes block scales into the .pte as bf16, rounded to nearest
    (backends/xnnpack/operators/node_visitor.py, ``scale.to(torch.bfloat16)``): eight bits
    of mantissa, so up to 0.39 percent either way. Solving against a scale the file cannot
    store would leave the codes matched to a weight the runtime never sees.
    """
    return scale.detach().to(_torch().bfloat16).to(_torch().float32)


def fake_act(x, bits: int = 8):
    """The per-token asymmetric int8 rounding the delegate applies to activations."""
    torch = _torch()
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    lo = torch.clamp(x.amin(-1, keepdim=True), max=0.0)
    hi = torch.clamp(x.amax(-1, keepdim=True), min=0.0)
    scale = torch.clamp((hi - lo) / (qmax - qmin), min=torch.finfo(torch.float32).eps)
    zero = torch.clamp(qmin - torch.round(lo / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale) + zero, qmin, qmax)
    return (q - zero) * scale


def _torch():
    import torch

    return torch


def group_scale(w):
    """The symmetric int4 scale per row of one group, by squared error over the clip grid."""
    torch = _torch()
    amax = w.abs().amax(1, keepdim=True).clamp(min=1e-8)
    best = best_err = None
    for clip in CLIPS:
        scale = bf16_storage(amax * clip / INT4_HALF_RANGE)
        err = ((torch.clamp(torch.round(w / scale), -8, 7) * scale - w) ** 2).sum(1, keepdim=True)
        if best is None:
            best, best_err = scale, err
        else:
            better = err < best_err
            best = torch.where(better, scale, best)
            best_err = torch.where(better, err, best_err)
    return best


def solve_weight(weight, hessian, group: int = GROUP, damp: float = 0.01, block: int = 128):
    """GPTQ for one linear: returns int4 codes and their per-group scales."""
    torch = _torch()
    device = weight.device
    w = weight.clone().to(torch.float32)
    out_f, in_f = w.shape
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    if in_f % group:
        raise ValueError(f"in_features={in_f} is not divisible by group={group}")

    # Factored in fp64 on the CPU, on a copy: the caller's Hessian is not touched.
    h = hessian.detach().cpu().to(torch.float64).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead.to(device)] = 0
    h = h + torch.eye(in_f, dtype=h.dtype) * (damp * torch.diag(h).mean())
    chol = torch.linalg.cholesky(h)
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True).to(torch.float32).to(device)

    codes = torch.zeros(out_f, in_f, dtype=torch.int8, device=device)
    scales = torch.zeros(out_f, in_f // group, device=device)
    for b0 in range(0, in_f, block):
        b1 = min(b0 + block, in_f)
        wb = w[:, b0:b1].clone()
        eb = torch.zeros_like(wb)
        hb = hinv[b0:b1, b0:b1]
        for i in range(b1 - b0):
            col = b0 + i
            if col % group == 0:
                end = col + group
                local_end = min(end, b1)
                wg = wb[:, i : i + (local_end - col)]
                if local_end < end:
                    wg = torch.cat((wg, w[:, local_end:end]), dim=1)
                scales[:, col // group] = group_scale(wg)[:, 0]
            scale = scales[:, col // group]
            column = wb[:, i]
            q = torch.clamp(torch.round(column / scale), -8, 7)
            codes[:, col] = q.to(torch.int8)
            err = (column - q * scale) / hb[i, i]
            wb[:, i:] -= err.unsqueeze(1) * hb[i, i:].unsqueeze(0)
            eb[:, i] = err
        w[:, b1:] -= eb @ hinv[b0:b1, b1:]
    return codes.cpu(), scales.cpu()


def dequantise(codes, scales, group: int = GROUP):
    """The weight the delegate will actually compute with."""
    torch = _torch()
    rows = codes.shape[0]
    return (codes.to(torch.float32).reshape(rows, -1, group) * scales.to(torch.float32).unsqueeze(-1)).reshape(
        codes.shape
    )


def eager_model(model_class: str, params: dict, checkpoint: Path, work_dir: Path, max_seq_len: int = 2048):
    """The module ``export_llm`` traces, in fp32 with no KV cache.

    Built through the same entry point the exporter uses, so the linears solved here are the
    ones lowered later; a different construction would solve a different model.
    """
    torch = _torch()
    from executorch.examples.models.llama.model import Llama2Model
    from executorch.extension.llm.export.config.llm_config import LlmConfig, ModelType

    if model_class.startswith("lfm2"):
        # Without this every calibration row would start on the previous row's convolution
        # state, so each Hessian would be taken from inputs no real prompt ever produces.
        # The exported graph carries the same fix, so the solve matches what ships.
        from pipeline import lfm2_state

        lfm2_state.apply()

    params_path = work_dir / "solve-params.json"
    params_path.write_text(json.dumps(params))
    cfg = LlmConfig()
    cfg.base.model_class = ModelType(model_class)
    cfg.base.params = str(params_path)
    cfg.base.checkpoint = str(checkpoint)
    cfg.model.use_kv_cache = False
    cfg.model.use_sdpa_with_kv_cache = False
    cfg.model.dtype_override = "fp32"
    cfg.export.max_seq_length = max_seq_len
    cfg.export.max_context_length = max_seq_len
    return Llama2Model(cfg).get_eager_model().to(torch.float32).eval()


def int4_linears(model) -> dict:
    """Every linear the 8da4w recipe puts at int4: the layers, and the output head.

    The head is included because leaving it out is a silent choice: it is the tied
    embedding's copy and a ninth of LFM2.5's weights, and every solve before 2026-09-18
    stopped at the last layer without saying so (finding 29).
    """
    from torch import nn

    found = {}
    for index, layer in enumerate(model.layers):
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                found[f"layers.{index}.{name}"] = module
    head = getattr(model, "output", None)
    if isinstance(head, nn.Linear):
        found["output"] = head
    return found


def teacher(src_dir: Path):
    """The fp32 Hugging Face model as a greedy continuation function, and the model itself.

    The replies are what GPTQ is asked to protect, so they come from the unquantised model,
    and from Hugging Face's own implementation rather than the module ``export_llm`` traces:
    ``generate`` carries a KV cache, and the app-shaped rows are ~530 tokens of head before
    the question starts. Re-forwarding the whole sequence for every token instead cost 31 of
    the 41 minutes the solve job spent on an 8-vCPU ARM runner with rows a tenth as long.
    The caller frees it before the eager model is built, so only one copy is ever resident.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(src_dir), dtype=torch.float32).eval()
    # Several families set eos_token_id to a list of stop tokens. generate() takes a list
    # there but pad_token_id has to be one integer, so the first is the one to pad with.
    eos = model.config.eos_token_id
    pad = eos[0] if isinstance(eos, list | tuple) else eos

    def generate(ids: list[int], max_new_tokens: int) -> list[int]:
        with torch.no_grad():
            out = model.generate(
                torch.tensor([ids], dtype=torch.long),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad,
            )
        return out[0, len(ids) :].tolist()

    return model, generate


def solve_model(model, rows, damp: float = 0.01, act_quant: bool = True, log=print) -> dict:
    """Solve every int4 linear, layer by layer, and return ``name -> {qdata, scale}``.

    ``rows`` is ``calibration.sequences``' output: ``(token ids, first position to count)``.
    The app-shaped rows all open with the same ~530-token head, which is in the sequence
    because without it the questions would not carry the app's distribution, but is counted
    in one row only. Counted in every row it would be four fifths of the Hessian's mass and
    the questions and replies -- the positions the tool-call decision lives in -- would be
    what the solve rounds away first.
    """
    torch = _torch()
    from torch import nn

    class _Stop(Exception):
        pass

    captured: list = []

    def grab(module, args, kwargs):
        captured.append((args, kwargs))
        raise _Stop

    keeps = [keep for _, keep in rows]
    handle = model.layers[0].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        for ids, _ in rows:
            try:
                model(torch.tensor([ids], dtype=torch.long))
            except _Stop:
                pass
    handle.remove()
    hidden = [call[0][0] for call in captured]
    # Which row the layer loop below is replaying, so the hooks know where to start counting.
    row = [0]

    def counted(x):
        """``x`` from ``keeps[row]`` on, flattened to (positions, width)."""
        start = keeps[row[0]]
        return (x[:, start:, :] if x.dim() == 3 else x[start:]).reshape(-1, x.shape[-1])

    codes: dict = {}
    started = time.time()
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            prefix = f"layers.{index}."
            linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
            hess = {n: torch.zeros(m.weight.shape[1], m.weight.shape[1]) for n, m in linears.items()}
            counts = dict.fromkeys(linears, 0)
            hooks = []
            for name, module in linears.items():

                def accumulate(mod, args, name=name, hess=hess, counts=counts):
                    x = counted(args[0])
                    if x.shape[0] == 0:
                        return
                    if act_quant:
                        x = fake_act(x)
                    hess[name] += x.T @ x
                    counts[name] += x.shape[0]

                hooks.append(module.register_forward_pre_hook(accumulate))
            for i, (args, kwargs) in enumerate(captured):
                row[0] = i
                layer(hidden[i], *args[1:], **kwargs)
            for hook in hooks:
                hook.remove()

            for name, module in linears.items():
                q, s = solve_weight(module.weight.data, hess[name] / max(counts[name], 1), damp=damp)
                codes[prefix + name] = {"qdata": q, "scale": s}
                module.weight.data = dequantise(q, s)
                if act_quant:
                    module.register_forward_pre_hook(lambda mod, args: (fake_act(args[0]),))
            del hess
            for i, (args, kwargs) in enumerate(captured):
                out = layer(hidden[i], *args[1:], **kwargs)
                hidden[i] = out[0] if isinstance(out, tuple | list) else out
            log(f"layer {index} solved, {time.time() - started:.0f}s")

        head = getattr(model, "output", None)
        if isinstance(head, nn.Linear):
            width = head.weight.shape[1]
            hess_head = torch.zeros(width, width)
            positions = 0
            for index, state in enumerate(hidden):
                row[0] = index
                x = counted(model.norm(state))
                if x.shape[0] == 0:
                    continue
                if act_quant:
                    x = fake_act(x)
                hess_head += x.T @ x
                positions += x.shape[0]
            q, s = solve_weight(head.weight.data, hess_head / max(positions, 1), damp=damp)
            codes["output"] = {"qdata": q, "scale": s}
            log(f"output head solved, {time.time() - started:.0f}s")
    return codes


def run(model_id: str, revision: str, out: Path, work_dir: Path, damp: float = 0.01) -> dict:
    """Solve one model's int4 codes and write them to ``out``.

    One solve serves every window: the codes are per linear, and the window changes only the
    KV cache and the masks, not the weights. So this runs once per model and the export
    matrix reuses its artifact, which is also why it is a separate job.
    """
    import torch

    from pipeline import calibration, convert, eligibility, families, gate, hub, settings
    from pipeline.exporting import ExportError

    cfg = settings.load()
    source = hub.fetch(model_id, revision)
    verdict = eligibility.evaluate(source, cfg)
    if verdict.reasons:
        raise ExportError(f"{model_id} is not eligible: {'; '.join(verdict.reasons)}")
    if verdict.backends["xnnpack"] is not None:
        raise ExportError(f"{model_id} cannot be exported to xnnpack: {verdict.backends['xnnpack']}")

    family = families.family_for(source.config)
    plan = families.xnnpack_plan(family, source.config)
    src_dir = work_dir / "source"
    checkpoint = work_dir / "checkpoint" / "consolidated.pth"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> solve {model_id}@{source.sha[:12]}: family {family.key}, class {plan.model_class}")
    hub.download(source, src_dir)
    print("==> converting checkpoint")
    convert.convert(src_dir, checkpoint, plan.converter, families.text_config(source.config))

    print("==> calibration")
    import gc

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(src_dir))
    fp32, generate = teacher(src_dir)
    rows = calibration.sequences(generate, tokenizer)
    # While the unquantised model is still loaded: what it would decide on the gate's
    # prompts, which is what the exported file has to agree with before anything publishes.
    print("==> gate reference")
    out.parent.mkdir(parents=True, exist_ok=True)
    reference = gate.reference(fp32, tokenizer)
    (out.parent / "gate.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(f"    fp32 is confident on {reference['confident']} of {len(reference['rows'])} gate rows")
    del fp32, generate
    gc.collect()

    print("==> building the eager model")
    model = eager_model(plan.model_class, plan.params, checkpoint, work_dir)

    print(f"==> solving {len(int4_linears(model))} linears")
    started = time.time()
    codes = solve_model(model, rows, damp=damp)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codes, out)
    seconds = time.time() - started
    print(f"==> wrote {out} ({len(codes)} linears) in {seconds:.0f}s")
    return {
        "model_id": model_id,
        "revision": source.sha,
        "linears": len(codes),
        "calibration_rows": len(rows),
        "calibration_positions": sum(len(ids) - keep for ids, keep in rows),
        "calibration_app_rows": len(calibration.SEARCH_ROWS) + len(calibration.KNOWN_ROWS),
        "seconds": round(seconds, 1),
        "damp": damp,
        "gate_rows": len(reference["rows"]),
        "gate_confident_rows": reference["confident"],
    }
