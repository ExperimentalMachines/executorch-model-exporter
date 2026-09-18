# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

exe-expo exports small dense open-weight LLMs (≤4B, no MoE) from Hugging Face to ExecuTorch `.pte` files for the
[openweights](https://github.com/alpharomercoma/openweights) Android app, running entirely on hosted CI runners
(Blacksmith `blacksmith-8vcpu-ubuntu-2404-arm`: 8 vCPU, 24 GB RAM + 24 GiB swap this repo adds). `docs/PLAN.md` holds the decisions, measured costs, phase status and known
limits. Read it before changing sizing, backends or naming, and update it when a decision or measurement changes.

## Commands

```sh
pip install -r requirements/dev.txt      # enough for lint + all tests except tests/test_convert.py
# test_convert.py is skipped unless torch + safetensors are present (CI installs them from the CPU index):
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu && pip install safetensors==0.8.0 numpy

ruff check . && ruff format --check .    # CI runs both; line length 120
pytest -q                                # all tests
pytest tests/test_naming.py::test_app_model_name_matches_the_kotlin_examples   # one test
pytest -k app_family                     # by keyword (parametrized tests)

python -m pipeline plan Qwen/Qwen3-1.7B  # eligibility, backends and window choice (Hub metadata only, no download)
```

A real export (`python -m pipeline export-xnnpack <model> --out out --work work`, or `export-qnn <model> --soc SM8650`)
needs Linux x86_64 and the pinned toolchain in `requirements/export-*.txt`. torch and torchvision must be installed
from the CPU index first, as in `.github/actions/setup-export/action.yml`. Other subcommands: `watch`, `publish-hf`,
`publish-release`, `summary` (see `pipeline/__main__.py`).

## Architecture

`pipeline/` is one package driven by `python -m pipeline <command>`. Each GitHub workflow calls these subcommands:

- **`watch-hf.yml`** (hourly) → `watch.py`: lists each watched org's newest repos, checks every unseen one (by name
  first with no network, then by config), and runs `gh workflow run` for each backend's export workflow. State is
  `seen.json` on the orphan **`state` branch**. Dispatch is staged (`watch.STAGES`: XNNPACK for every model, then Vulkan, then QNN, then
  MediaTek; the next stage starts when the previous has nothing queued or running) and idempotent (a run in flight for
  the model counts as dispatched); `requeue` puts cancelled dispatches back. The first run only seeds the state; existing models are exported via the
  `backfill` input. `watch.WORKFLOWS` maps backend → workflow.
- **`solve` job in `export-xnnpack.yml`** → `solve.py`: GPTQ for the int4 codes, once per model because the codes are
  per linear and the window changes only the KV cache and the masks. Calibration is `calibration.py`: committed
  prompts in the model's own chat template, each continued by the fp32 model's own greedy reply. The codes reach every
  window's export as an artifact; `export_with_codes.py` injects them into torchao's tensors before lowering and also
  carries the LFM2 state fix (`lfm2_state.py`), which the solve applies too.
- **`export-xnnpack.yml`** / **`export-vulkan.yml`** → `export_xnnpack.py` (`backend="xnnpack"|"vulkan"`, same recipe, XNNPACK or Vulkan delegate; Vulkan gets the structural check): `hub.fetch` → `eligibility.evaluate` → `sizing.choose_context` →
  download → `convert.py` (HF safetensors → ExecuTorch checkpoint layout) → generated `params.json` + `export_llm`
  YAML → subprocess `executorch.extension.llm.export.export_llm` → `smoke.py` (greedy generation through the wheel's
  `TextLLMRunner`, the same C++ runner the app uses) → `export-report-<window>.json` + `config.json`.
- **`export-qnn.yml`** (matrix job per chip in `qnn.socs`) → `export_qnn.py`: runs ExecuTorch's own Qualcomm script
  (`executorch.examples.qualcomm.oss_scripts.llama.llama --compile_only`) as a subprocess with `QNN_SDK_ROOT`/
  `LD_LIBRARY_PATH` pre-set (`qnn_env`). The script always downloads `main`, so a `--revision` that isn't `main`'s
  current commit is refused. No HTP runtime on the host, so the check is structural: the program's flatbuffer is
  parsed (`delegates()`) and each decoder method must delegate to `QnnBackend`.
- **`publish.py`**: commits one backend folder to `experimentalmachines/<name>-ExecuTorch`, pinned to the parent
  revision and retried on HTTP 412, because backends publish concurrently. It regenerates the README from every backend's
  `export-report-<window>.json` (`manifest.py`), then attaches files under 2 GiB to a GitHub release.

`export-report-<window>.json` is the contract between export, publish and summary; `manifest.backend_config` derives
the app's `config.json` from a folder's reports. **`export-mtk.yml`** → `export_mtk.py` runs ExecuTorch's
`examples/mediatek` scripts in a separate Python 3.10 venv with MediaTek's wheels (see the module docstring).

### Key modules

- `families.py`: HF architecture → family, and per-family `XnnpackPlan` (model_class, params, converter). For XNNPACK,
  `export_llm`'s `model_class` only picks the example dir; **the params file defines the architecture**, so new
  sizes/finetunes of a known architecture work without ExecuTorch listing them. QNN, by contrast, only exports
  checkpoints hard-coded in ExecuTorch's registry: `QNN_DECODERS`, `QNN_PARAMS`, `QNN_META_CHECKPOINT` (Llama 3.2 uses
  Meta's `original/` checkpoint).
- `eligibility.py`: `Verdict` with skip `reasons` plus per-backend `None`/reason. `name_reasons` is the network-free
  pre-check the watcher uses.
- `naming.py`: output names, plus **Python ports of the app's Kotlin name rules** (`app_family`, `app_backend`,
  `app_model_name`, `app_size_hint`). The app infers chat template and backend from names, so every generated name must
  pass `check_app_rules`.
- `sizing.py`: the memory model. Every tier in `context_tiers` is exported (one workflow job per window); the phone
  estimate (.pte + fp32 KV cache + overhead vs `device_budget_bytes`) is recorded per file as `fits_phone_budget`, and
  only the host estimate (export peak grows with window² from per-layer causal masks) is a gate: over it the export
  raises `SkipExport` (CLI exit 4, workflow "skipped"). Calibrated against probe runs in PLAN.md.
- `publish.py`: one run publishes one window. `superseded()` decides which existing files that window replaces; the
  folder's `config.json` (`variants[]`, one per window) and the repo README are regenerated from every
  `export-report-<window>.json` in the repo. Never delete another window's files.
- `settings.py`: frozen `Settings` loaded (cached) from `config/pipeline.yaml` + `config/versions.env`.

## Constraints that aren't obvious from the code

- **Version lock:** `EXECUTORCH_VERSION` in `config/versions.env` must equal the executorch-android AAR the app ships.
  A newer exporter can emit methods the older runtime lacks. `tests/test_versions.py` checks the pip pins against it;
  bump pins, `versions.env` and `QAIRT_VERSION` together.
- **App-imposed naming rules** (from the app source, see PLAN.md): the output repo name must never contain `xnnpack`,
  must keep the size (`1.7B`), and repo name + file stem must contain an app family token and none of `vl`, `vision`,
  `coder`, `guard`, `qwen35`.
- **Tokenizer only at the repo root.** A tokenizer inside any backend folder stops the app lending the root one to the
  other folders; `publish_hf` refuses it.
- `third_party/executorch/` holds params files copied verbatim from ExecuTorch v1.4.0 (the wheel doesn't ship them) for
  the QNN registry. Refresh them when the ExecuTorch pin moves.
- QNN `.pte` metadata always has BOS 1 / EOS 2 (hard-coded upstream); stop tokens must come from the tokenizer.
- Upstream licenses are copied verbatim into published repos; vendor SDKs (QAIRT, NeuroPilot) are downloaded at run
  time and never re-hosted. Before adding MediaTek (phase 4), write `THIRD_PARTY_NOTICES.md` from the SDK's license text.

## CI and supply chain

Actions are pinned to commit SHAs with a `# vX.Y.Z` comment; `.github/dependabot.yml` bumps them. Every requirement
is pinned (`tests/test_versions.py` enforces it, and that the backend files agree on shared pins except
`transformers`, which is deliberately ExecuTorch's example pin for QNN). Workflow inputs reach shell only through
`env:` variables; keep it that way.

## Tests

`tests/fixtures/*.config.json` are real HF `config.json` files, and `et-*.params.json` are ExecuTorch's own params for
the same models. `tests/test_convert.py` and `tests/test_qnn.py` skip without torch/executorch; CI installs both (CPU
index), and locally a scratch venv with `torch==2.13.0 executorch==1.4.0` runs them on macOS too. Tests check generated params against them and against the vendored `third_party` copies. Build a
`SourceModel` with the `source` fixture / `conftest.make_source`; `TOTAL_PARAMS` holds real Hub parameter counts. The
watcher takes injectable `lister`/`fetch`/`dispatcher` callables, so tests never touch the network.

## Conventions

Commit subjects are short and scoped by area (`QNN export: …`, `PLAN: …`) and explain the reason, not just the change.
Code comments cite the upstream source they depend on (ExecuTorch file, app Kotlin class) and measured numbers; keep
doing that.

Never put a `Claude-Session:` trailer or any claude.ai session link in a commit message, PR, release, model card or
anything else published: this repo is public and its history has been rewritten to remove them.
