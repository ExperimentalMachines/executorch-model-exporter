"""Memory arithmetic for choosing an export's context window.

Units are bytes throughout. Every estimate here is for a full-attention decoder with an
fp32 KV cache, which is what the XNNPACK recipe exports (docs/PLAN.md, "Context auto-fit").
"""

from __future__ import annotations

from dataclasses import dataclass

FP32_BYTES = 4
# 8da4w with group size 32: half a byte per 4-bit weight plus 2 bytes of scale per group
# of 32 (2 / 32 = 0.0625 bytes per weight).
LINEAR_BYTES_PER_PARAM_8DA4W_G32 = 0.5 + 2 / 32
# int8 per-channel embedding table: one byte per weight (one scale per row is negligible).
EMBEDDING_BYTES_PER_PARAM_INT8 = 1.0
# RoPE cos/sin tables grow the file with the window: Qwen3-0.6B measured 496,570,368 B at
# 2k and 525,932,032 B at 16k, 2,048 B per extra token = 4 x head_dim (128) x 4 bytes.
ROPE_TABLE_BYTES_PER_TOKEN_PER_HEAD_DIM = 4 * FP32_BYTES
# Norms, graph and delegate headers. With the terms above the estimate before this margin
# was within 0.3% of all three measured files (SmolLM2-135M 2k, Qwen3-0.6B 2k and 16k).
PTE_OVERHEAD_FACTOR = 1.01
# Export peak RSS = fp32 weights + KV cache + causal masks + a fixed cost for torch.export
# and lowering. Calibrated on the probe runs (ubuntu-latest, 16.8 GB RAM): Qwen3-0.6B peaked
# at 5,836,587,008 B at 2k and 15,781,117,952 B at 16k; the fixed part is ~2.24 GB, and the
# weights count once (LFM2.5-2.6B: 12.2 GB peak for 10.4 GB of fp32 weights).
EXPORT_PEAK_WEIGHT_MULTIPLE = 1.0
EXPORT_FIXED_OVERHEAD_BYTES = 2_500_000_000


@dataclass(frozen=True)
class Architecture:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    dim: int
    intermediate: int
    total_params: int  # HF safetensors total: whatever the checkpoint stores
    tied_embeddings: bool
    # How many layers actually attend. Equal to n_layers for a plain transformer, smaller
    # for a hybrid: LFM2's short-convolution layers keep a few columns of state instead of
    # a KV cache, and build no causal mask. Counting them as attention layers overstates
    # the export peak enough to refuse a window that fits, and the masks are the dominant
    # term at 32k.
    attention_layers: int | None = None

    @property
    def attending(self) -> int:
        return self.n_layers if self.attention_layers is None else self.attention_layers

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.dim

    @property
    def linear_params(self) -> int:
        """Every linear the exported model has, counted from the architecture.

        Not derived from total_params: some tied checkpoints store lm_head anyway (Qwen3-0.6B)
        and some do not (SmolLM2), while the export always gives the output projection its
        own quantized copy."""
        q_out = self.n_heads * self.head_dim
        kv_out = self.n_kv_heads * self.head_dim
        attention = self.dim * q_out + 2 * self.dim * kv_out + q_out * self.dim
        mlp = 3 * self.dim * self.intermediate
        return self.n_layers * (attention + mlp) + self.vocab_size * self.dim


def kv_cache_bytes(arch: Architecture, context: int) -> int:
    """K and V for every attending layer, every KV head, every position, fp32."""
    return arch.attending * 2 * arch.n_kv_heads * arch.head_dim * context * FP32_BYTES


def pte_bytes_estimate(arch: Architecture, context: int) -> int:
    raw = (
        arch.embedding_params * EMBEDDING_BYTES_PER_PARAM_INT8
        + arch.linear_params * LINEAR_BYTES_PER_PARAM_8DA4W_G32
        + context * arch.head_dim * ROPE_TABLE_BYTES_PER_TOKEN_PER_HEAD_DIM
    )
    return int(raw * PTE_OVERHEAD_FACTOR)


def device_resident_bytes(arch: Architecture, context: int, overhead: int) -> int:
    return pte_bytes_estimate(arch, context) + kv_cache_bytes(arch, context) + overhead


def causal_mask_bytes(arch: Architecture, context: int) -> int:
    """Every AttentionMHA layer builds its own window x window bool mask at construction
    (ExecuTorch 1.4.0 examples/models/llama/attention.py). The custom SDPA op does not keep
    it in the .pte, but the eager model holds all of them while exporting. A hybrid's
    convolution layers are not AttentionMHA and build none."""
    return arch.attending * context * context


# How far the estimate below runs under what the runner actually uses. Measured against
# `host.peak_in_use_bytes` in 24 published export reports: it under-predicts in 18 of them,
# by a median of 1.13x and a worst case of 1.70x, and the worst cases are the widest
# windows -- Qwen3-1.7B at 16k was estimated at 20.4 GiB and used 34.7, finishing with 34 MB
# of memory to spare on a runner with 23.4 GiB of RAM and 24 GiB of swap.
#
# Without this factor the estimate for that model at 32k is 44.9 GiB against a 46.5 GiB
# budget, so the window was attempted and the runner was killed mid-export -- twice, in runs
# 35413952925 and 35420508341, each time reported only as "the runner has received a
# shutdown signal". With it, that window is refused as too wide and recorded as a skip,
# while every window that has ever succeeded still fits: the widest of those, LFM2.5-2.6B at
# 32k, is estimated at 21.8 GiB and measured 36.5.
HOST_PEAK_HEADROOM = 1.7


def export_peak_bytes(arch: Architecture, context: int) -> int:
    weights = (arch.embedding_params + arch.linear_params) * FP32_BYTES
    return int(
        weights * EXPORT_PEAK_WEIGHT_MULTIPLE
        + kv_cache_bytes(arch, context)
        + causal_mask_bytes(arch, context)
        + EXPORT_FIXED_OVERHEAD_BYTES
    )


@dataclass(frozen=True)
class WindowChoice:
    context: int | None
    reason: str
    table: tuple[dict, ...]


def choose_context(
    arch: Architecture,
    tiers: tuple[int, ...],
    device_budget: int,
    runtime_overhead: int,
    host_budget: int | None,
) -> WindowChoice:
    """Largest tier whose phone residency and host export peak both fit."""
    rows = []
    chosen = None
    for context in sorted(tiers, reverse=True):
        resident = device_resident_bytes(arch, context, runtime_overhead)
        peak = export_peak_bytes(arch, context)
        fits_device = resident <= device_budget
        fits_host = host_budget is None or peak * HOST_PEAK_HEADROOM <= host_budget
        rows.append(
            {
                "context": context,
                "kv_cache_bytes": kv_cache_bytes(arch, context),
                "device_resident_bytes": resident,
                "export_peak_bytes": peak,
                "fits_device": fits_device,
                "fits_host": fits_host,
            }
        )
        if chosen is None and fits_device and fits_host:
            chosen = context
    if chosen is None:
        smallest = rows[-1]
        reason = (
            f"no window fits: at {smallest['context']} tokens the phone would hold "
            f"{smallest['device_resident_bytes']:,} B (budget {device_budget:,} B) and the "
            f"export would peak near {smallest['export_peak_bytes']:,} B"
        )
        return WindowChoice(None, reason, tuple(rows))
    return WindowChoice(chosen, f"largest window within budgets: {chosen}", tuple(rows))
