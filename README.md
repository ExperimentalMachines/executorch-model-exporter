# exe-expo

Exports small open-weight LLMs from Hugging Face to ExecuTorch `.pte` files for the
[openweights](https://github.com/alpharomercoma/openweights) Android app, on standard
hosted CI runners. Design and decisions: [docs/PLAN.md](docs/PLAN.md).

| Backend | Status |
|---|---|
| XNNPACK (CPU) | Qwen3, Qwen2.5, Llama 3.2, SmolLM2 |
| Vulkan (GPU) | the same models and recipe as XNNPACK, through ExecuTorch's Vulkan delegate; checked structurally (no GPU on the runner) |
| Qualcomm QNN (SM8750; SM8650 optional) | checkpoints in ExecuTorch 1.4.0's Qualcomm registry (Qwen3, Qwen2.5 base, Gemma 3 1B, SmolLM2 135M, SmolLM3 3B, Llama 3.2) |
| MediaTek NeuroPilot (MT6989; MT6991 optional) | Qwen3, Qwen2.5 (phase 4, first export pending); Llama 3.2 and Gemma 3 not validated yet |
| Samsung Exynos (ENN) | waiting for an LLM path in ExecuTorch (1.4.0 ships the delegate with CNN examples only) |
| HF watcher (auto-dispatch) | hourly; XNNPACK for every model first, then Vulkan, then QNN, then MediaTek; state on the `state` branch |

## Running an export

Actions → **Export XNNPACK** (or **Export Vulkan**, **Export QNN**, **Export MediaTek**) → Run workflow, with a
model id such as `Qwen/Qwen3-1.7B`. One job per context window (2k, 4k, 8k, 16k, 32k; the
`contexts` input narrows it) and, for the NPU backends, per chip. Each job exports, checks
the result (XNNPACK: a smoke test with ExecuTorch's `TextLLMRunner`, the runner the app
uses; NPU backends: a structural check, as there is no host NPU runtime), uploads an
artifact, and publishes its window into `experimentalmachines/<model>-ExecuTorch` on
Hugging Face beside the other windows, plus a GitHub release. A window the runner cannot
build (export memory grows with the square of the window) is reported as skipped, not failed.

The QNN and MediaTek workflows download Qualcomm's and MediaTek's SDKs from their publishers
on each run, which accepts their license terms: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Repository secrets:

- `HF_TOKEN`: a Hugging Face token with write access to the output org
  (`hub.org` in [config/pipeline.yaml](config/pipeline.yaml)). Its account must have
  accepted the licenses of gated source models (Llama, Gemma).

**Watch Hugging Face** runs hourly and dispatches in stages: every model's XNNPACK
exports first, then Vulkan, then Qualcomm, then MediaTek; a stage starts once the previous one has no
run queued or running. Its first run records the models that already
exist without exporting them; to export existing ones, run it with `backfill` set to a
comma-separated list of model ids (or run **Export XNNPACK** directly). `dry_run` reports
what it would do without dispatching or saving state.

**Probe runner** exports one small model at a list of windows (default 2k and 16k) without publishing, to
measure what the runner really has and calibrate the sizing estimates.

## Findings

Everything learned from running the pipeline is logged with its evidence (run links, log
excerpts, report fields, ExecuTorch source lines, reproducers) in
[docs/research/](docs/research/README.md). The headline so far:

**Where export time goes** ([study](docs/research/export-bottlenecks.md)). A Qualcomm
export of Qwen3-0.6B takes 2,434-5,987 s on the hosted runner (measured, 2k and 4k), and two
stages are 87-93% of it:

| Run | Calibration loop | HTP compile | Export |
|---|---|---|---|
| SM8650, 2k | 1,761 s (48.4%) | 1,538 s (42.3%) | 3,637 s |
| SM8750, 2k | 936 s (38.5%) | 1,172 s (48.2%) | 2,434 s |
| SM8750, 4k | 3,674 s (61.4%) | 1,905 s (31.8%) | 5,987 s |

- The calibration loop is one pass over a window-length sequence, plus ExecuTorch's SeqMSE
  search, which its recipe enables for Qwen3-0.6B and Llama-3.2-1B only: 101 evaluations of
  each of the 197 conv layers over the whole sequence. By operation count that is about 99%
  of the pass (estimate; the logs time the loop as a whole).
- The HTP compile is Qualcomm's SDK building the prompt and decode graphs; CPU-bound.
- Runners vary: identical 2k calibrations took 1,761 s and 936 s. The 2k to 4k growth on one
  chip (3.92x) is not yet explained; reports now record the CPU model and peak swap in use.
- XNNPACK exports take about 10-12 minutes and are limited by memory, not time.
- The first MediaTek export (Qwen3-0.6B, MT6991, 512-token cache) took 1,695 s: calibration
  736 s (43%), NeuroPilot lowering 472 s (28%), over 4 chunks. Its limit is calibration
  memory: the window it can build is set by the runner's RAM plus swap, not by time.

## Locally

```sh
pip install -r requirements/dev.txt
python -m pipeline plan Qwen/Qwen3-1.7B        # eligibility, backends and the per-window sizing table, no download
pytest
```

A full export needs Linux x86_64 (the executorch wheel's LLM runner is not built for
Windows); see the install steps in
[.github/actions/setup-export/action.yml](.github/actions/setup-export/action.yml), then
`python -m pipeline export-xnnpack <model> --context 4096 --out out --work work` (one window per
invocation; without `--context`, the largest the host can build). Exit codes: 0 exported, 2
failed, 4 skipped (this host cannot build the window).

## Layout

- `config/pipeline.yaml`: output org, watched orgs, size limits, context tiers, phone memory budget, quantization recipe.
- `config/versions.env`: ExecuTorch release every backend must match (the app's runtime).
- `pipeline/`: eligibility, family recipes, checkpoint conversion, export, smoke test, publishing.
- `tests/`: unit tests; `tests/fixtures` holds real HF `config.json` files and ExecuTorch's own params files.
