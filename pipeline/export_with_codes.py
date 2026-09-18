"""Run ``export_llm`` with solved int4 codes in place of its own rounding.

    python -m pipeline.export_with_codes --config <export_llm.yaml> --codes <codes.pt>

A separate entry point, invoked as a subprocess like plain ``export_llm``, so the exporter
keeps measuring a child's peak RSS and an out-of-memory export still fails as a process
rather than taking the job down.

How the codes get in. ``export_llama`` quantises through
``examples/models/llama/source_transformation/quantize.quantize``; this replaces that
function for the ``8da4w`` mode only, lets torchao build its tensors as usual, and then
overwrites each one's ``qdata`` and ``scale`` with the solved values. The result is the
same tensor subclass, the same packing and the same delegate: only the numbers differ. Any
int4 tensor the codes do not cover is named in the log rather than left to look solved,
because a silently rounded linear is indistinguishable from a solved one in the file
(the output head was exactly that until 2026-09-18).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def inject(model, codes: dict, log=print) -> tuple[int, list[str]]:
    """Copy solved codes into the torchao tensors, and report what was left rounded."""
    injected = 0
    left: list[str] = []
    for fqn, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "qdata"):
            continue
        entry = codes.get(fqn)
        if entry is None:
            left.append(fqn)
            continue
        qdata, scale = entry["qdata"], entry["scale"]
        if tuple(qdata.shape) != tuple(weight.qdata.shape):
            raise ValueError(f"{fqn}: codes are {tuple(qdata.shape)}, the tensor is {tuple(weight.qdata.shape)}")
        if tuple(scale.shape) != tuple(weight.scale.shape):
            raise ValueError(f"{fqn}: scales are {tuple(scale.shape)}, the tensor is {tuple(weight.scale.shape)}")
        # XNNPACK's blockwise int4 format carries no sign on the scale and no code outside
        # -8..7; a file that breaks either loads and computes something else.
        if not bool((scale > 0).all()):
            raise ValueError(f"{fqn}: a scale is not positive")
        if int(qdata.min()) < -8 or int(qdata.max()) > 7:
            raise ValueError(f"{fqn}: codes are off the int4 grid")
        weight.qdata.copy_(qdata.to(weight.qdata.dtype))
        weight.scale.copy_(scale.to(weight.scale.dtype))
        if hasattr(weight, "zero_point") and weight.zero_point is not None:
            weight.zero_point.zero_()
        injected += 1
    log(f"injected calibrated codes into {injected} linears")
    if left:
        log(f"left to round to nearest: {sorted(left)}")
    return injected, left


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the export_llm YAML the exporter wrote")
    parser.add_argument("--codes", required=True, help="codes.pt from the solve stage")
    args = parser.parse_args(argv)

    import torch
    from executorch.examples.models.llama.source_transformation import quantize as quantize_module

    # weights_only: the codes file is an artifact passed between CI jobs, and it holds
    # nothing but tensors, so there is no reason to let it unpickle arbitrary objects.
    codes = torch.load(args.codes, map_location="cpu", weights_only=True)
    upstream = quantize_module.quantize

    def quantize(model, qmode, *rest, **kwargs):
        quantised = upstream(model, qmode, *rest, **kwargs)
        if qmode != "8da4w":
            return quantised
        inject(quantised, codes)
        return quantised

    quantize_module.quantize = quantize

    from executorch.extension.llm.export.export_llm import main as export_llm_main

    sys.argv = ["export_llm", "--config", str(Path(args.config))]
    export_llm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
