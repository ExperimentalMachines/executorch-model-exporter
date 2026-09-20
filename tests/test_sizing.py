import math

from conftest import TOTAL_PARAMS, hf_config, load_json

from pipeline import families, sizing

TIERS = (32768, 16384, 8192, 4096, 2048)
BUDGET = 5_000_000_000
OVERHEAD = 500_000_000


def arch(model_id):
    return families.architecture(hf_config(model_id), TOTAL_PARAMS[model_id])


def test_kv_cache_matches_the_plan_literal():
    # docs/PLAN.md: 28 x 2 x 8 x 128 x 32,768 x 4 = 7,516,192,768 bytes.
    qwen = arch("Qwen/Qwen3-1.7B")
    assert sizing.kv_cache_bytes(qwen, 1) == 229_376
    assert sizing.kv_cache_bytes(qwen, 32768) == 7_516_192_768


def test_kv_cache_qwen3_4b():
    # 36 layers x 2 x 8 KV heads x 128 x 4 bytes = 294,912 per token.
    assert sizing.kv_cache_bytes(arch("Qwen/Qwen3-4B"), 1) == 294_912
    assert sizing.kv_cache_bytes(arch("Qwen/Qwen3-4B"), 32768) == 9_663_676_416


def test_linear_params_are_counted_from_the_architecture():
    # Qwen3-0.6B per layer: q 1,024x2,048 + k,v 2 x 1,024x1,024 + o 2,048x1,024
    # + mlp 3 x 1,024x3,072 = 15,728,640; x 28 = 440,401,920; + output 151,936 x 1,024.
    qwen = arch("Qwen/Qwen3-0.6B")
    assert qwen.linear_params == 440_401_920 + 155_582_464
    assert qwen.embedding_params == 155_582_464
    # Its checkpoint stores lm_head despite tying, so the HF total counts the table twice.
    # Norm weights: 28 x (2 x 1,024 + 2 x 128 q/k norms) + 1,024 final = 65,536.
    assert qwen.total_params == 440_401_920 + 2 * 155_582_464 + 65_536


def test_pte_estimate_matches_the_measured_exports():
    qwen = arch("Qwen/Qwen3-0.6B")
    # 155,582,464 + 595,984,384 x 0.5625 + 2,048 x 128 x 16 = 495,017,984; x 1.01
    # Measured 496,570,368 B (probe run 34696579580).
    assert sizing.pte_bytes_estimate(qwen, 2048) == 499_968_163
    # + 16,384 x 128 x 16 instead: 524,378,112; x 1.01. Measured 525,932,032 B (run 34697093999).
    assert sizing.pte_bytes_estimate(qwen, 16384) == 529_621_893
    smol = families.architecture(load_json("smollm2-135m.config.json"), 134_515_008)
    # 28,311,552 + 134,479,872 x 0.5625 + 2,048 x 64 x 16 = 106,053,632; x 1.01.
    # Measured 106,018,048 B (local Docker run).
    assert sizing.pte_bytes_estimate(smol, 2048) == 107_114_168


def test_resident_bytes_around_the_qwen3_1_7b_boundary():
    # .pte at 8k: (311,164,928 + 1,720,451,072 x 0.5625 + 8,192 x 128 x 16) x 1.01
    qwen = arch("Qwen/Qwen3-1.7B")
    assert sizing.pte_bytes_estimate(qwen, 8192) == 1_308_652_830
    assert sizing.device_resident_bytes(qwen, 8192, OVERHEAD) == 3_687_701_022
    assert sizing.device_resident_bytes(qwen, 16384, OVERHEAD) == 5_583_694_202


def test_export_peak_matches_the_probe():
    # fp32 weights (155,582,464 + 595,984,384) x 4 = 3,006,267,392, + KV cache + masks + fixed.
    qwen = arch("Qwen/Qwen3-0.6B")
    # Measured peak RSS 5,836,587,008 B (probe run 34696579580): estimate 4.4% over.
    assert sizing.export_peak_bytes(qwen, 2048) == 3_006_267_392 + 469_762_048 + 117_440_512 + 2_500_000_000
    assert sizing.export_peak_bytes(qwen, 2048) == 6_093_469_952
    # Measured peak RSS 15,781,117,952 B at 16k (run 34697093999), with part of it in swap.
    assert sizing.export_peak_bytes(qwen, 16384) == 3_006_267_392 + 3_758_096_384 + 7_516_192_768 + 2_500_000_000


def test_window_choice_per_model():
    expected = {
        "Qwen/Qwen3-0.6B": 16384,  # 529,621,893 + 3,758,096,384 + 500,000,000 = 4,787,718,277
        "Qwen/Qwen3-1.7B": 8192,
        "Qwen/Qwen3-4B": 4096,  # 2,686,471,495 + 1,207,959,552 + 500,000,000 = 4,394,431,047
        "meta-llama/Llama-3.2-1B-Instruct": 32768,  # 1,001,243,607 + 2,147,483,648 + 500,000,000
    }
    for model_id, context in expected.items():
        choice = sizing.choose_context(arch(model_id), TIERS, BUDGET, OVERHEAD, None)
        assert choice.context == context, model_id


def test_causal_masks_dominate_export_memory_at_32k():
    # The probe's Qwen3-0.6B 32k export killed a 16.8 GB + 24 GB swap runner: 28 layers of
    # 32,768 x 32,768 one-byte masks is 30,064,771,072 bytes before weights or KV cache.
    qwen = arch("Qwen/Qwen3-0.6B")
    assert sizing.causal_mask_bytes(qwen, 32768) == 30_064_771_072
    host_budget = 16_766_414_848 + 25_769_799_680 - 1_000_000_000
    assert sizing.export_peak_bytes(qwen, 32768) > host_budget
    assert sizing.export_peak_bytes(qwen, 16384) < host_budget


def test_window_choice_respects_host_budget():
    qwen = arch("Qwen/Qwen3-0.6B")
    unconstrained = sizing.choose_context(qwen, TIERS, 10**12, OVERHEAD, None)
    assert unconstrained.context == 32768
    # The budget has to cover the estimate plus the headroom the estimate is known to run
    # under by, which is what choose_context compares against.
    peak_at_16k = math.ceil(sizing.export_peak_bytes(qwen, 16384) * sizing.HOST_PEAK_HEADROOM)
    limited = sizing.choose_context(qwen, TIERS, 10**12, OVERHEAD, peak_at_16k)
    assert limited.context == 16384


def test_no_window_fits():
    choice = sizing.choose_context(arch("Qwen/Qwen3-4B"), TIERS, 1_000_000_000, OVERHEAD, None)
    assert choice.context is None
    assert "no window fits" in choice.reason
    assert [row["context"] for row in choice.table] == list(TIERS)


# The Blacksmith 8vcpu ARM runner, as its own export reports record it: 23.4 GiB of RAM and
# 24 GiB of swap, less the 1 GB reserve host_budget keeps.
RUNNER_BUDGET = int(23.4 * 2**30) + int(24.0 * 2**30) - 1_000_000_000


def allowed_windows(model_id):
    choice = sizing.choose_context(arch(model_id), TIERS, BUDGET, OVERHEAD, RUNNER_BUDGET)
    return sorted(row["context"] for row in choice.table if row["fits_host"])


def test_a_window_that_killed_the_runner_twice_is_refused():
    """Qwen3-1.7B at 32k was attempted and killed the runner in runs 35413952925 and
    35420508341, reported only as "the runner has received a shutdown signal".

    It attends on all 28 layers, so its per-layer causal masks grow with the window where a
    hybrid's do not: the raw estimate is 44.9 GiB against a 46.5 GiB budget, which fit, and
    the measured figure would have been far higher -- the same model at 16k was estimated at
    20.4 GiB and used 34.7.
    """
    assert allowed_windows("Qwen/Qwen3-1.7B") == [2048, 4096, 8192, 16384]


def test_the_widest_window_that_did_succeed_is_still_allowed():
    """The headroom has to refuse the window that died without refusing one that worked.

    Qwen3-1.7B at 16k is the tightest export that has finished on this runner: estimated
    20.4 GiB, measured 34.7, with 34 MB of memory left at the low point. It is in the list
    above, and 32k is not.
    """
    assert 16384 in allowed_windows("Qwen/Qwen3-1.7B")
    # And the smaller model of the same family keeps exactly the windows it actually has:
    # experimentalmachines/Qwen3-0.6B-ExecuTorch publishes four .pte files, 2k to 16k, and
    # never had a 32k one. The headroom agrees with what the fleet really contains.
    assert allowed_windows("Qwen/Qwen3-0.6B") == [2048, 4096, 8192, 16384]


def test_the_headroom_is_the_worst_ratio_actually_measured():
    # 24 published reports carry both the estimate and host.peak_in_use_bytes. The estimate
    # runs under in 18 of them, by a median of 1.13x and a worst case of 1.70x.
    assert sizing.HOST_PEAK_HEADROOM >= 1.7
