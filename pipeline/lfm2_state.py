"""LFM2's short convolution forgets nothing, so a published export must clear it in the graph.

Upstream: upstream keeps `conv_state` in a buffer and
never clears it, and the runner's reset() only rewinds the position (ExecuTorch 1.4.0,
examples/models/lfm2/short_conv.py; extension/llm/runner/text_llm_runner.cpp). A new
prompt therefore starts on the last two columns of the previous one. Measured
2026-09-17 on the fp32 eager module, second prompt onward: the tool-call token falls
from 0.94 to 0.68 (Rome) and from 0.92 to 0.38 (Jerusalem); with the state cleared the
module equals Hugging Face to three decimals on all sixteen rows.

The fix is in the graph, because a published .pte cannot be told to zero a buffer: the
state is multiplied by min(input_pos, 1), so a prefill that starts at position zero reads
zeros whatever the buffer holds, and a continuation reads the state as before.

An export carrying it advertises ``get_state_reset_at_zero``, which is how a host tells a
fixed file from one that needs reopening between prompts. Upstream's own Qwen3.5 definition
already does the equivalent for its recurrent state; only LFM2 was left without it.
"""

import torch


def apply():
    from executorch.examples.models.lfm2 import short_conv as sc

    def conv_forward(self, x: torch.Tensor, input_pos: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seqlen, dim = x.size()
        assert batch_size == 1, "batch_size must be 1"
        B = self.B_proj(x).transpose(-1, -2)
        C = self.C_proj(x).transpose(-1, -2)
        x = self.x_proj(x).transpose(-1, -2)
        Bx = B * x
        if input_pos is None:
            state = torch.zeros_like(self.conv_state)
        else:
            # min(position, 1), not (position > 0): the same 0 or 1, from a clamp every
            # delegate implements. The comparison lowered on XNNPACK, and on a Galaxy S25
            # Ultra the Vulkan delegate, whose partitioner had accepted it, aborted at load
            # with "Missing operator: aten.gt.Scalar" (2026-09-18).
            keep = torch.clamp(input_pos.reshape(-1)[:1], max=1).to(self.conv_state.dtype).view(1, 1, 1)
            state = self.conv_state * keep
        Bx = torch.cat([state, Bx], dim=-1)
        new_conv_state = Bx[..., -(self.L_cache - 1) :]
        with torch.no_grad():
            self.conv_state.copy_(new_conv_state)
        conv_out = self.conv(Bx)[..., : x.size(-1)]
        y = (C * conv_out).transpose(-1, -2).contiguous()
        return self.out_proj(y)

    def block_forward(self, x, freqs_cos=None, freqs_sin=None, attn_options=None):
        pos = attn_options.get("input_pos") if attn_options else None
        h = self.conv.forward(self.attention_norm(x), pos)
        h = x + h
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, None

    sc.ShortConv.forward = conv_forward
    sc.ShortConvBlock.forward = block_forward
