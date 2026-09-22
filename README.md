# exe-expo

Exports small open-weight LLMs from Hugging Face to ExecuTorch `.pte` files for the
[openweights](https://github.com/alpharomercoma/openweights) Android app, on
[Blacksmith](https://www.blacksmith.sh)'s hosted CI runners: ARM for the GPTQ solve and the
XNNPACK export, x86 for the runner check before anything publishes, and x86 at 8, 16 or 32
vCPUs for MediaTek, sized per context window. Design and decisions: [docs/PLAN.md](docs/PLAN.md).

What has shipped to [`experimentalmachines`](https://huggingface.co/experimentalmachines) on
Hugging Face, and what is built but not published:

| Backend | Status |
|---|---|
| XNNPACK (CPU) | **Published**: LFM2.5 1.2B and 2.6B (and their heretic finetunes), Qwen3, Qwen2.5, Llama 3.2, SmolLM2, at every window from 2k to 32k the runner could build. Each window is now compared with its fp32 model before it publishes. Twelve models were published before that gate existed and are being re-checked by **Audit published**; the two audited so far, SmolLM2-135M and 360M, score 0.19 and 0.22 on its prompts against about 0.58 for their fp32 models |
| MediaTek NeuroPilot (MT6991) | **Published**: LFM2.5 1.2B and 2.6B (and their heretic finetunes) at MediaTek's default 512-token window. The sizing now puts 2k to 32k within the 32-vCPU tier; a 4k export has been built and answered correctly on a Dimensity 9400, and nothing above 512 is published yet. Checked structurally on the runner, which has no NPU |
| Vulkan (GPU) | Workflow in place, same models and recipe as XNNPACK through ExecuTorch's Vulkan delegate, checked structurally (no GPU on the runner). Nothing published yet |
| Qualcomm QNN (SM8750; SM8650 optional) | Workflow in place for the checkpoints in ExecuTorch 1.4.0's Qualcomm registry (Qwen3, Qwen2.5 base, Gemma 3 1B, SmolLM2 135M, SmolLM3 3B, Llama 3.2). Nothing published yet |
| Samsung Exynos (ENN) | Waiting for an LLM path in ExecuTorch (1.4.0 ships the delegate with CNN examples only) |
| HF watcher (auto-dispatch) | Run by hand: its hourly schedule has been off since 2026-09-19. Dispatches XNNPACK for every model first, then Vulkan, then QNN, then MediaTek; state on the `state` branch |

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

**Watch Hugging Face** runs when dispatched by hand (its hourly schedule has been off since
2026-09-19) and dispatches in stages: every model's XNNPACK
exports first, then Vulkan, then Qualcomm, then MediaTek; a stage starts once the previous one has no
run queued or running. Its first run records the models that already
exist without exporting them; to export existing ones, run it with `backfill` set to a
comma-separated list of model ids (or run **Export XNNPACK** directly). `dry_run` reports
what it would do without dispatching or saving state.

**Probe runner** exports one small model at a list of windows (default 2k and 16k) without publishing, to
measure what the runner really has and calibrate the sizing estimates.

**Audit published** downloads files already on Hugging Face and compares what each decides with
its fp32 source on the gate's prompts. It only reads; it publishes and deletes nothing.

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
  736 s (43%), NeuroPilot lowering 472 s (28%), over 4 chunks. Its limit was calibration
  memory, which held every prompt at once and pinned every window at 512. Holding one prompt at
  a time moved the rest to disk, so the runner for each window is now picked by RAM and disk
  together, and the sizing puts 2k to 32k within the 32-vCPU tier (4k built so far).

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

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

That covers the pipeline, not what it produces. Each model it publishes keeps its upstream
model's license, whose files are copied verbatim into that model's Hugging Face repo. The
Qualcomm and MediaTek SDKs are downloaded from their publishers at run time under their own
terms and are never redistributed here: see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
