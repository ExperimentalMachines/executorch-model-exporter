# exe-expo: Hugging Face → ExecuTorch export pipeline

Watches Hugging Face for new small dense LLMs and exports them to ExecuTorch `.pte` for
the [openweights](https://github.com/alpharomercoma/openweights) Android app and its
benchmarker: every ExecuTorch Android backend, XNNPACK (CPU), Vulkan (GPU), Qualcomm QNN
(Snapdragon HTP), MediaTek NeuroPilot (Dimensity APU) and, when ExecuTorch gains an LLM path
for it, Samsung Exynos (ENN). iOS is deferred.

## Decisions

| Topic | Decision |
|---|---|
| Models | Dense (no MoE) text LLMs of the 4B class and smaller (size in the name ≤ 4B; real count < 4.5B, since Qwen3-4B is 4,022,468,096), instruct and base, original bf16/fp16 weights only |
| Watched orgs | `Qwen`, `google`, `meta-llama`, `HuggingFaceTB`, each for its own families |
| Trigger | Hourly watcher; eligible models are dispatched in stages, XNNPACK for every model first, then Vulkan, then Qualcomm, then MediaTek (a stage waits until the previous one has nothing queued or running) |
| First run | Seeds state without exporting; existing models are backfilled by manual dispatch |
| Runners | Blacksmith, one workflow run per backend and one job per window. Ubuntu ARM throughout except the smoke test: solve and export on `blacksmith-8vcpu-ubuntu-2404-arm`, orchestration and CI on the 2 vCPU ARM size, and a `verify` job on x86 that runs the C++ runner (which fails on aarch64) and publishes only once the file has answered (finding 30) |
| Chips | The benchmark devices ("Devices" below): QNN SM8750 (Snapdragon 8 Elite), MediaTek MT6989 (Dimensity 9300+); SM8650 and MT6991 can be added in `config/pipeline.yaml` |
| Outputs | Hugging Face Hub, GitHub Releases (files ≤ 2 GiB), Actions artifacts |
| HF layout | One repo per model, backend folders; NPU exports published even though the app can't load them yet |
| Context window | Every window of 2k/4k/8k/16k/32k, per model and backend, one workflow job each; the app and the benchmarker decide what fits a device. A window the runner cannot build is skipped and reported, not failed |
| Runtime pin | ExecuTorch **1.4.0** everywhere; QAIRT **2.37.0** (what `executorch-android-qnn:1.4.0` depends on) |
| Vendor SDKs | Downloaded from the vendor at run time, cached only in this repo's Actions cache, never re-hosted |

## Hugging Face repo layout

```
experimentalmachines/Qwen3-1.7B-ExecuTorch
├── README.md                 # generated from every export-report-*.json in the repo
├── LICENSE…                  # upstream license files, copied verbatim
├── tokenizer.json            # root only: nested weights borrow it (HuggingFaceClient.kt)
├── xnnpack/Qwen3-1.7B-8da4w-{2k,4k,8k,16k}.pte   # one file per window the runner could build
├── xnnpack/config.json                             # variants[]: every window, smallest first
├── xnnpack/export-report-{2k,4k,8k,16k}.json       # the full record of each export
├── vulkan/Qwen3-1.7B-vulkan-8w-{2k,…}.pte, config.json, export-report-*.json
├── qnn/sm8750/…                                    # same shape per chip
└── mtk/mt6989/…                                    # chunks per window, one shared embedding table
```

Each window is its own workflow job and publishes on its own: `publish.publish_hf` replaces
only the files of its own window in its folder (`superseded`), regenerates the folder's
`config.json` from every `export-report-*.json` there, and the README from every report in
the repo, pinned to the parent commit and retried on conflict. Every variant carries
`context` and `fits_phone_budget` (the sizing estimate against the 5 GB budget, `None` for
the NPU backends) so a consumer can pick without downloading.

Constraints taken from the app source:

- Repo carries the `executorch` tag (Discover searches `filter=executorch`).
- Size stays in the repo name (`1.7B`): the size filter reads it.
- Repo name must not contain `xnnpack`: `CompiledBackend.of(repoId + path)` checks `xnnpack`
  first, which would make every QNN/MTK file in the repo read as XNNPACK.
- Normalised `repo name + file stem` must contain a family token the app has a template
  for (`qwen3`, `qwen25`, `smollm2`, `smollm3`, `llama32`, `phi4mini`, `gemma3`, `lfm25`)
  and none of `vl`, `vision`, `coder`, `guard`, `qwen35`. `tests/test_naming.py` ports
  these rules and checks every generated name.
- `config.json` next to each `.pte`, in the `variants[].methods` form Discover reads.

## Context windows and the sizing model

ExecuTorch fixes the window at export and allocates the full fp32 KV cache at load. The
openweights window matrix found Qwen3-1.7B at 32k resident at 6.1-8.5 GB and killed by
Samsung's and MIUI's 6 GB memory guards. Every window is exported anyway (the benchmark
matrix needs them, and the app filters), and the sizing model below tells consumers what
fits and tells the exporter what the runner can build:

- KV cache bytes = `n_layers × 2 × n_kv_heads × head_dim × window × 4`.
  Qwen3-1.7B at 32k: 28 × 2 × 8 × 128 × 32,768 × 4 = 7,516,192,768 bytes.
- Resident on the phone ≈ `.pte` + KV cache + 0.5 GB (measured in the app's research).
- `.pte` estimate for 8da4w/g32 + int8 embeddings = (`embedding params × 1 B + linear
  params × 0.5625 B + window × head_dim × 16 B` of RoPE tables) × 1.01, with linear
  params counted from the architecture (the output projection always gets its own 4-bit
  copy). Within 1% of every measured file.
- Export peak ≈ fp32 weights + KV cache + `n_layers × window²` bytes of causal masks +
  2.5 GB.
- `fits_phone_budget` = resident ≤ `device_budget_bytes` (default 5.0 GB), recorded per
  file. Estimated export peak > runner RAM + swap is the one gate: the job exits 4 and the
  workflow records the window as skipped (Qwen3 at 32k on hosted runners, see Known limits).

## Backends

### XNNPACK (phase 1)

`export_llm` from the pip wheel, the recipe the app was measured with: 8-bit dynamic
activations and 4-bit weights in groups of 32, int8 per-channel embeddings, XNNPACK with
extended ops, prefill chunk 2048, fp32 KV cache, the family's BOS/EOS ids in metadata.
`export_llm`'s `model_class` only chooses the example directory; the params file defines
the architecture, so it is generated from the HF `config.json`, and the HF safetensors are
converted to ExecuTorch's checkpoint layout by `pipeline/convert.py`.

| Family | HF architecture | XNNPACK in 1.4.0 |
|---|---|---|
| Qwen3 | `Qwen3ForCausalLM` | yes |
| Qwen2.5 | `Qwen2ForCausalLM` | yes |
| Llama 3.2, SmolLM2 | `LlamaForCausalLM` | yes (q/k un-permuted for Meta RoPE) |
| Gemma 3, SmolLM3 | `Gemma3ForCausalLM`, `SmolLM3ForCausalLM` | not yet: not in `export_llm`'s model list; needs a validated path (optimum-executorch or params support) |

Smoke test: the Linux wheel ships `TextLLMRunner`, the same C++ runner the app calls
through `LlmModule`. Greedy generation of a short prompt must produce non-degenerate text
(and "Paris" for models ≥ 500M parameters); the `.pte`'s metadata methods are read back
and written to `config.json`. The wheel's runner only links portable and XNNPACK kernels,
so `portable_lib`, `custom_ops` and `kernels.quantized` are imported first to register
`llama::custom_sdpa`, `update_cache` and `embedding_byte` (the app's AAR links them).

First end-to-end run (local Docker, 8 GB, SmolLM2-135M-Instruct at 2k): 106,018,048-byte
`.pte` (estimate 112,383,432), export peak RSS 2,748,440,576 B, 31 min of mostly
single-threaded lowering, reply "The capital of France is Paris." Gated repos the token
cannot read are reported as a skip reason, not a crash.

### Vulkan (phase 6)

`export_llm` with `backend.vulkan` instead of `backend.xnnpack` and the same recipe: the
Vulkan delegate runs torchao's `Int8DynamicActivationIntxWeight` linears (8da4w) and the
int8 embeddings (`docs/source/backends/vulkan/vulkan-quantization.md`). One module serves
both (`export_xnnpack.run(backend="vulkan")`); files are named `<name>-vulkan-8da4w-<w>.pte`
so the app's `CompiledBackend.of` reads "vulkan". The Linux wheel's runner has no Vulkan
kernels and the runner no GPU, so the check is structural (`smoke.structural_check`:
`forward` delegates to `VulkanBackend`); quality and speed are measured on the devices.

### QNN (phase 3)

No source build: the executorch 1.4.0 Linux x86_64 wheel ships the Qualcomm backend
(`PyQnnManagerAdaptor`, `libqnn_executorch_backend.so`) and downloads QAIRT 2.37.0.250724
plus a libc++ into `~/.cache/executorch/qnn` on first import; the workflow caches that
directory per QAIRT version. `export-qnn.yml` runs one job per chip in `qnn.socs`, each
calling ExecuTorch's own script in compile-only mode:

`python -m executorch.examples.qualcomm.oss_scripts.llama.llama --decoder_model <key>
--soc_model <SoC> --compile_only --model_mode hybrid --max_seq_len 2048 --max_context_len
2048 --prefill_ar_len 128 --calib_tasks wikitext --calib_limit 1`

- Only checkpoints in the script's registry (`SUPPORTED_LLM_MODELS`) can be exported; from
  the watched orgs that is Qwen3-0.6B/1.7B, Qwen2.5-0.5B/1.5B (base), gemma-3-1b-it,
  SmolLM2-135M-Instruct, SmolLM3-3B and Llama-3.2-1B/3B-Instruct (`families.QNN_DECODERS`).
  Gemma 3 and SmolLM3 are QNN-only for now (no XNNPACK path yet).
- Each registry entry carries its own quantization recipe; the script downloads the weights
  from the entry's repo at `main`. Llama 3.2 entries have no repo, so Meta's original
  checkpoint, params and tokenizer (`original/` in meta-llama's repos) are passed in.
- The other entries point at params `.json` files in ExecuTorch's source tree, which the
  wheel does not package; copies from v1.4.0 live in `third_party/executorch/` and are passed
  with `--params` (`families.QNN_PARAMS`). Tests check them against the HF configs.
- The wheel's import-time SDK setup edits `LD_LIBRARY_PATH` after the loader has read it, so
  QNN cannot open `libQnnSystem.so` by name. The script is started with `QNN_SDK_ROOT` and
  `LD_LIBRARY_PATH` (SDK libs + soname links to the wheel's libc++) already set
  (`export_qnn.qnn_env`), ExecuTorch's documented manual setup.
- Calibration dependencies are ExecuTorch's example pins: `transformers==5.0.0rc1`,
  `datasets==3.6.0`, `lm_eval==0.4.5` (`requirements/export-qnn.txt`).
- No host runtime for HTP binaries, so the check is structural: the program loads, has
  `prefill_forward` and `kv_forward`, and each of those graphs delegates to `QnnBackend`
  (read from the program's flatbuffer, not by searching the file for the name).
- The script downloads its registry entry's repo at `main`, so the run refuses a `revision`
  that is not what `main` currently points at, rather than recording one commit and
  compiling another.
- The window is fixed at compile time: `qnn.max_context_len` (2048 to start), or the
  workflow's `context` input (`--context`) for one run.
- QAIRT's license: `THIRD_PARTY_NOTICES.md`.

### MediaTek (phase 4)

`export-mtk.yml` runs one job per chip in `mtk.socs` (MT6989 = DX3, MT6991 = DX4):

- **SDK:** NeuroPilot Express 8.0.8 (build 20250925), from MediaTek's download table,
  downloaded on every run, checked against `NEUROPILOT_SDK_SHA256`, and deleted once
  `mtk_converter` 8.13.0 and `mtk_neuron` 8.2.23 are installed. 8.2.23 is the version
  ExecuTorch 1.4.0 documents; the build 20250327 archive its CI installs has 8.2.19, which
  lacks `mtk_neuron.extract_shared_data` that the LLM path calls (docs/research, finding
  19). The license (read 2026-09-13, accepted for ExperimentalMachines, byte-identical in
  both builds) is summarised in `THIRD_PARTY_NOTICES.md`; the agreement is marked MediaTek
  Confidential, so it is paraphrased, not copied.
- **Two Python environments:** `mtk_converter` is cp310-only and `examples/mediatek`
  imports transformers 4.x internals, which cap `huggingface_hub` below the 1.x the pipeline
  uses. The pipeline keeps its own environment (`requirements/export-mtk.txt`) and drives a
  Python 3.10 virtualenv (`requirements/mtk-tools.txt` + MediaTek's wheels) as `MTK_PYTHON`.
  The export script starts with `mtk_neuron/lib` first on `LD_LIBRARY_PATH`
  (`export_mtk.tool_env`): the wheel's `libextract_shared.so` needs the `libc++.so.1`
  bundled beside it but has no RPATH (docs/research, finding 21).
- **Scripts:** the LLM export scripts are in ExecuTorch's source, not the wheel; the
  workflow sparse-checks-out `examples/mediatek` at `EXECUTORCH_COMMIT`.
- **Recipe** (MediaTek's own, `shell_scripts/export_qwen.sh`): A16W4, the model cut into up
  to 4 chunks of equal layer counts, a 128-token prompt graph and a one-token generation
  graph over a 512-token cache, calibrated on MediaTek's `alpaca.txt` prompts (9, up to 9
  generated tokens each) in the family's chat template. The cache is MediaTek's default
  because calibration keeps a full fp32 KV cache per prompt and step: the first run, at
  2048, exhausted the runner's 16.8 GB + 24 GB swap while preparing calibration inputs
  (estimate 85 GB, `export_mtk.calibration_bytes`, now checked before any download).
- **Patch:** `third_party/executorch/patches/mediatek-calibration-as-arrays.patch` makes
  MediaTek's calibration read its prepared inputs back as arrays instead of nested Python
  lists; without it the second run needed ~35 min per prompt and could not finish
  (docs/research, finding 17). Values are unchanged; `export_mtk` refuses an unpatched
  script.
- **Families:** the scripts build the model from `config.json`'s `model_type`, so any Qwen3
  or Qwen2.5 size works, not a fixed list. Llama 3.2 (the scripts read
  `rope_scaling['type']`, its config has `rope_type: llama3`), SmolLM2 (tokenizer class) and
  Gemma 3 (`gemma3_text` vs `gemma3`) wait for validation.
- **Output** per chip: the chunk `.pte` files, the fp32 token embedding table the runner
  reads from disk, and `config.json` with the flags MediaTek's LLM runner
  (`examples/mediatek/executor_runner`) needs. They do not run on `TextLLMRunner`.
- **Check:** structural; every chunk loads, has its two methods, and delegates to
  `NeuropilotBackend`.
- **First export** (2026-09-13, not published): Qwen3-0.6B, MT6991, 512-token cache, all 9
  prompts, structural check passed. 4 chunks (3 x 86 MB, 165 MB with the output layer) plus
  a 622 MB fp32 embedding table, 1.05 GB in all. Export 1,695 s, calibration 43% of it.
  Memory in use reached up to ~29.0 GB (16.5 GB of RAM + 12.4 GB of swap), above the 21.1 GB
  `calibration_bytes` estimate (docs/research, findings 22-23).
- **Published** (2026-09-13): Qwen3-0.6B for MT6989 at 512, `mtk/mt6989/` and release
  `Qwen3-0.6B-mtk-mt6989-512-c1899de`; export 1,343 s (docs/research, finding 26).
- **Windows:** `export-mtk.yml` with no `contexts` (as the watcher dispatches it) runs the
  `context_tiers` plus `mtk.cache_size` 512 (finding 24). The memory gate
  (`calibration_bytes`: 6 B per parameter + 2.4 x the calibration tensors, fitted to the
  measured 29.6 GB peak at 512) skips Qwen3-0.6B at 2,048 and up, so for now only 512 builds
  (finding 25).

## Phases

0. **Probe** (`probe-runner.yml`, done 2026-09-12): `ubuntu-latest` on this public repo is
   4 vCPU (AMD EPYC 9V74), 16,766,414,848 B RAM, one 160 GB NVMe (~103 GB free after
   cleanup, no `/mnt`), plus the 24 GiB swap file. Qwen3-0.6B exports:

   | Window | `.pte` | Peak RSS | Export | Smoke |
   |---|---|---|---|---|
   | 2k | 496,570,368 B | 5,836,587,008 B | 600 s | "Paris", 83 tok/s decode |
   | 16k | 525,932,032 B | 15,781,117,952 B | 721 s | "Paris", 83 tok/s decode |
   | 32k | — | runner killed (causal masks) | — | — |

   `pipeline/sizing.py` is calibrated on these: estimates within 1% of every measured
   `.pte` and 4-7% above the measured export peaks.
1. **XNNPACK end to end** (`export-xnnpack.yml`, done 2026-09-12): first publish
   [experimentalmachines/Qwen3-0.6B-ExecuTorch](https://huggingface.co/experimentalmachines/Qwen3-0.6B-ExecuTorch)
   (16k window, smoke test "Paris") and GitHub release `Qwen3-0.6B-xnnpack-c1899de`.
2. **Watcher** (`watch-hf.yml`, hourly at :17, plus manual dispatch with `backfill`, `dry_run`
   and `requeue` inputs; dispatches in stages, XNNPACK → Vulkan → QNN → MediaTek, `watch.STAGES`).
   Lists every repo of each org (`limit_per_org` 1,500); a repo not yet in
   `seen.json` (on the `state` branch) is checked once, by name first and then by config,
   and each org only for its own families (`org_families`). Eligible models go to
   `export-xnnpack.yml` at the revision seen, at most 6 runs per watcher run. A dry run over
   the 40 newest real repos (2026-09-12) skipped all of them correctly: Qwen3.8 multimodal
   and FP8, Qwen3-ASR, Llama 4, Llama Guard, SmolLM3 (no XNNPACK recipe yet), and
   HuggingFaceTB's GSM8K fine-tune of Qwen3 (not HuggingFaceTB's own family).
3. **QNN** (`export-qnn.yml`; the watcher dispatches it for registry checkpoints). First
   publish 2026-09-12: Qwen3-0.6B at 2k into the same repo (`qnn/sm8650/`, `qnn/sm8750/`)
   and releases `Qwen3-0.6B-qnn-<chip>-c1899de`, both passing the structural check:

   | Chip | `.pte` | Peak RSS | Export |
   |---|---|---|---|
   | SM8650 | 665,070,592 B | 15,657,545,728 B | 3,637 s (calibration + quantize 2,042 s, compile 1,538 s) |
   | SM8750 | 664,296,448 B | 15,644,954,624 B | 2,434 s |

   A 4k probe (`context` input, SM8750, not published) decides whether 4k becomes the
   default window.
4. **MediaTek** (`export-mtk.yml`, built 2026-09-13; first export pending).
5. **Window matrix** (2026-09-13): every tier per model and backend, one job each. First
   matrix publishes the same day: SmolLM2-135M/360M/1.7B at 2k, 4k, 8k and 16k into
   `experimentalmachines/<name>-ExecuTorch/xnnpack/` with merged `config.json`; 32k skipped
   on the runner as predicted. 22 models dispatched through the watcher's backfill
   (whole-org listing); the gated Llama 3.2 and Gemma 3 repos wait for the HF_TOKEN
   account to accept their licenses.

## Scope: every ExecuTorch Android backend, every supported family (2026-09-13)

Decision: export everything ExecuTorch 1.4.0 can build for Android, at every window, and
let the benchmarker decide downstream. iOS (CoreML, MPS) is deferred. Cells: **done** = the
pipeline exports it now; **todo** = ExecuTorch supports it and the pipeline does not yet;
**upstream** = not possible in ExecuTorch 1.4.0.

| Family (dense ≤ 4B) | XNNPACK (CPU) | Vulkan (GPU) | Qualcomm QNN | MediaTek | Samsung Exynos |
|---|---|---|---|---|---|
| Qwen3 0.6B / 1.7B / 4B (+Base, +2507) | done | todo | done: 0.6B, 1.7B (registry) | done | upstream |
| Qwen2.5 0.5B / 1.5B / 3B (+Instruct) | done | todo | done: 0.5B, 1.5B base (registry) | done | upstream |
| Llama 3.2 1B / 3B (+Instruct) | done | todo | done: 1B/3B-Instruct (registry) | todo (validate `llama.py`: rope_type, tokenizer) | upstream |
| SmolLM2 135M / 360M / 1.7B (+Instruct) | done | todo | done: 135M-Instruct (registry) | todo (validate, with Llama 3.2) | upstream |
| SmolLM3 3B | upstream | upstream | done | upstream | upstream |
| Gemma 3 1B | upstream | upstream | done: gemma-3-1b-it | todo (validate `gemma.py`: model_type gemma3_text) | upstream |
| Gemma 2 2B | upstream | upstream | todo | todo | upstream |
| Phi-4-mini 3.8B | todo (watch `microsoft`) | todo | todo | todo | upstream |
| LFM2.5 350M / 1.2B, LFM2 350M / 700M / 1.2B | todo (watch `LiquidAI`) | todo | upstream | upstream | upstream |
| Qwen3.5 0.8B / 2B / 4B | blocked: the app refuses `qwen35` names | — | upstream | upstream | upstream |
| GLM-edge 1.5B, Granite 3.3 2B | upstream | upstream | todo | upstream | upstream |

Sources, ExecuTorch v1.4.0: `extension/llm/export/config/llm_config.py` (`ModelType`,
`BackendConfig`), `examples/qualcomm/oss_scripts/llama/__init__.py` (`SUPPORTED_LLM_MODELS`),
`examples/mediatek/model_export_scripts/{llama,qwen,gemma,phi}.py` and
`aot_utils/llm_utils/utils.py` (`resolve_model_classes`: model_type llama, qwen2, qwen3,
phi3, phi4, gemma1/2/3), `examples/samsung` (CNN examples only; `backends/samsung` has the
`EnnBackend` delegate and chipset `E9955` = Exynos 2500 but no LLM path).

### Devices

| Device | Chip | Backend files it runs |
|---|---|---|
| Galaxy S25 Ultra | Snapdragon 8 Elite = QNN `SM8750` | `qnn/sm8750/`, plus XNNPACK and Vulkan |
| Galaxy Tab S10+ | Dimensity 9300+ = MediaTek `MT6989` (DX3) | `mtk/mt6989/`, plus XNNPACK and Vulkan |
| Pixel 10 family | Tensor G5 (no ExecuTorch NPU delegate) | XNNPACK and Vulkan only |
| Galaxy Z Flip7 | Exynos 2500 = ENN `E9955` (no LLM path in 1.4.0) | XNNPACK and Vulkan only |

NPU binaries are compiled per chip (QNN `--soc_model`, MediaTek `--platform`) and load only
on that chip, so the chip lists in `config/pipeline.yaml` (`qnn.socs`, `mtk.socs`) name the
benchmark devices; XNNPACK and Vulkan files run on any of them.

## Alignment with the master plan (2026-09-13)

The master diagram: openweights → llama.cpp and ExecuTorch runtimes; ExecuTorch on Android
with XNNPACK, Vulkan, Qualcomm, MediaTek and Samsung Exynos delegates, on iOS with XNNPACK and
CoreML; in GitHub Actions an **Exporter** (this repo, pull & push Hugging Face) feeding a
**Benchmarker** (pull from Hugging Face, stream to a device farm: Snapdragon 8 Elite, Tensor
G5, Dimensity 9300+, Exynos 2500; GSM8K, RetrievalQA, IFEval, PopQA, BFCL, FreshQA).

| Diagram element | Here | Note |
|---|---|---|
| Exporter, pull & push HF, GitHub Actions | done | plus the watcher, sizing and releases the diagram does not show |
| XNNPACK | done | Qwen3, Qwen2.5, Llama 3.2, SmolLM2; Gemma 3 and SmolLM3 are not in 1.4.0's `export_llm` model list |
| Qualcomm (SM8650, SM8750) | done | registry checkpoints only, one fixed window |
| MediaTek (MT6989, MT6991) | phase 4 | `mtk_converter` wheel, NeuroPilot SDK |
| Vulkan | absent | 1.4.0's `export_llm` has `backend.vulkan.enabled` and a `vulkan_8w` qmode, so it is an XNNPACK-shaped job (S/M); the app's `CompiledBackend` already reads `vulkan` |
| CoreML / iOS | absent | 1.4.0's `export_llm` has `backend.coreml` (ios 15-18, `coreml_*` qmodes); needs a macOS runner and iOS naming/tags in the app (M) |
| Samsung Exynos | absent | 1.4.0 ships `backends/samsung` (`EnnBackend`) but no LLM example for it (L, upstream-bound) |
| Tensor G5 | n/a | no ExecuTorch delegate for the Tensor TPU: it runs XNNPACK (or Vulkan) files |
| Benchmarker, device farm, benchmarks | absent | another component; what it needs from here is already published per file: `config.json` `variants[].methods` (window, prefill chunk, BOS/EOS), `qnn_sdk_version`, the tokenizer at the root |
| Several context lengths per model | done | one job per window in every export workflow (`contexts` input, default every tier); `publish_hf` keeps sibling windows and merges `variants[]`; the watcher's one dispatch per backend covers the matrix. 32k XNNPACK is out of reach for Qwen3-class models on hosted runners (Known limits) and is recorded as skipped |

## Known limits

- Export memory grows with the square of the window: every attention layer of ExecuTorch
  1.4.0's transformer builds its own window × window causal mask (not stored in the
  `.pte`). Qwen3-0.6B at 32k needs 30,064,771,072 bytes of masks alone and killed a
  16.8 GB + 24 GB swap runner; `sizing.export_peak_bytes` counts it and a forced window
  that cannot fit is refused.

- The watcher checks each repo once. A model whose weights change after it was seen
  (a new commit on `main`) is not re-exported automatically: most upstream commits touch
  cards and tokenizer configs, and every re-export is a multi-hour run. Re-export by hand
  with the `backfill` input or the export workflow.
- The watcher dispatches export runs before the workflow pushes `seen.json`. If that push
  fails, the next run dispatches the same models again; the per-model concurrency group
  queues the duplicate rather than running it alongside, and a repeat publish is
  idempotent, so the cost is runner time, not a broken repo.
- There is no memory or time model for QAIRT's compile, so the QNN matrix has no skip
  gate: measured calibration went from 936 s at 2k to 3,674 s at 4k on the same chip
  (docs/research/export-bottlenecks.md), and 16k/32k static graphs may hit the 6-hour
  job limit or the runner's memory. Those jobs fail on their own; the other windows of the
  same run publish regardless (`fail-fast: false`).
- QNN and MediaTek only export models ExecuTorch has hard-coded; a new family waits for an
  ExecuTorch release, and the app's AAR must move with it.
- QNN/MediaTek exports of 3-4B models may run out of memory or hit the 6-hour job limit
  on 16 GB runners; they fail independently of the other backends.
- NPU context windows are fixed at compile time and will be far smaller than 32k. Every
  decode step attends over the whole window, so a larger one slows every token, and the
  16-bit KV cache costs 114,688 B per token for Qwen3-0.6B (28 layers × K,V × 8 heads × 128
  × 2 B: 234,881,024 B at 2k).
- QNN `.pte` metadata carries `get_bos_id` 1 and `get_eos_id` 2 whatever the model:
  ExecuTorch 1.4.0 hard-codes them (`static_llama.py`), and its Qualcomm runner takes stop
  tokens from the tokenizer per family instead (`<|im_end|>` for Qwen). The app has to do
  the same when it loads QNN models.
- Upstream licenses differ (e.g. Qwen2.5-3B is under the Qwen Research License); cards and
  `LICENSE` files copy upstream's terms exactly.
