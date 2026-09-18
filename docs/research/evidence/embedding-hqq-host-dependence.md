# The same model, built on two runners, decodes different text

Collected 2026-09-19 from the published `export-report-*.json` of the organisation's
XNNPACK exports, before the base-model repositories were deleted. Each report records the
runner's `host.cpu_model` and the smoke test's `reply`, so the two can be crossed.

The smoke prompt is the same in every row: "What is the capital of France? Answer with one
word." The window is fixed at export and should not change what the model says; the
research note for the app measured the same codes at 2k, 8k and 32k giving identical
probabilities to four decimals when they were exported on one machine.

## SmolLM2-360M (base), run 34753837945

| Window | Runner CPU | Reply (first 56 characters) |
|---|---|---|
| 2,048 | Intel Xeon Platinum 85xx | ` Paris.\n\nIn the 19th century, the French Revolution occu` |
| 4,096 | AMD EPYC 7763 | ` Paris.\n\nFrance is a country in Western Europe.\n\nFrance ` |
| 8,192 | AMD EPYC 9V74 | ` Paris.\n\nIn 1960, the 20th century began.\n\nIn 1960, the ` |
| 16,384 | AMD EPYC 7763 | ` Paris.\n\nFrance is a country in Western Europe.\n\nFrance ` |

The 4k and 16k builds share a runner CPU and are identical. The other two windows landed on
different CPUs and each says something else. The reply tracks the CPU, not the window.

## SmolLM2-135M (base), run 34753835530

| Window | Runner CPU | Answered | Reply (first 56 characters) |
|---|---|---|---|
| 2,048 | Intel Xeon Platinum 83xx | yes | ` the city of Paris.\n\nThe city of Paris is the capital of` |
| 4,096 | AMD EPYC 7763 | yes | ` the city of Paris.\n\nThe city of Paris is the capital of` |
| 8,192 | AMD EPYC 9V74 | no | ` the capital of the country.\n\nThe capital of France is t` |
| 16,384 | AMD EPYC 9V74 | no | ` the capital of the country.\n\nThe capital of France is t` |

Both EPYC 9V74 builds fail to name the city; both other builds name it. Again the split is
by CPU, and here it crosses the smoke test's own pass criterion.

## Across the organisation

Every pair of windows inside one repository, 24 repositories, 97 XNNPACK programs:

| Pair | Count |
|---|---|
| same runner CPU, same reply | 50 |
| same runner CPU, different reply | 2 |
| different runner CPU, same reply | 62 |
| different runner CPU, different reply | 35 |

With the CPU held constant the reply repeats in 50 of 52 pairs; across CPUs it repeats in
62 of 97. Eleven of the 24 repositories had at least one window whose reply differed from
its siblings, most of them base models, which ramble and so amplify a small numerical
difference into different text.

## Cause

`embedding_quantize: "8,0"` reaches torchao as an int8 weight-only config whose scales come
from an HQQ search. That search is platform-dependent: the app's own research note measured
the same recipe exported on Apple silicon and on x86-64 differing in 16,930 bytes of
795,686,528, all inside the int8 token embedding, and up to 0.129 of probability on a single
row. The int4 linears were identical on both machines, and torchao's `affine` algorithm was
identical on both for both. A hosted runner is whatever GitHub gives the job, so a rebuild
is not reproducible.
