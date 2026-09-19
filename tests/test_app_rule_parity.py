"""The app's naming rules and this repository's port of them must agree.

`pipeline/naming.py` re-implements `PromptTemplates.forModel` from the openweights app
(core/common/.../model/PromptTemplate.kt), because the exporter has to refuse a name the
app would refuse and pick the family the app would pick. Nothing keeps the two in step, and
twice in two days they drifted and a failed export found it rather than a test:

* "qwen35" stayed in APP_EXCLUDED after the app gained Qwen35Template, so this repository
  skipped a family the app can run (2026-09-19);
* the family order here had "qwen3" before "qwen35", which is a prefix of it, so a Qwen3.5
  name would have been handed Qwen3's template.

This test reads the Kotlin and compares it to the Python. It is skipped, not failed, when
the app is not checked out beside this repository, so CI without it stays green -- but on a
machine that has both, drift is a test failure instead of a bad export.
"""

import re
from pathlib import Path

import pytest

from pipeline import naming

KOTLIN = Path.home() / (
    "mobile-inference/core/common/src/commonMain/kotlin/io/github/alpharomercoma/"
    "openweights/core/common/model/PromptTemplate.kt"
)


def kotlin_source() -> str:
    if not KOTLIN.exists():
        pytest.skip(f"the app is not checked out at {KOTLIN}")
    return KOTLIN.read_text(encoding="utf-8")


def test_the_family_tokens_and_their_order_match_the_app():
    source = kotlin_source()
    block = source[source.index("fun forModel(") : source.index("private val EXCLUDED")]
    # Each `"token" in name -> SomeTemplate` branch, in the order the app tries them.
    app_order = re.findall(r'"([a-z0-9]+)" in name ->', block)
    assert app_order, "could not read the family branches out of PromptTemplate.kt"
    assert list(naming.APP_FAMILY_TOKENS) == app_order, (
        f"app tries {app_order}, this repository tries {list(naming.APP_FAMILY_TOKENS)}"
    )


def test_the_exclusions_match_the_app():
    source = kotlin_source()
    line = re.search(r"private val EXCLUDED = listOf\(([^)]*)\)", source)
    assert line, "could not read EXCLUDED out of PromptTemplate.kt"
    app_excluded = re.findall(r'"([a-z0-9]+)"', line.group(1))
    assert sorted(naming.APP_EXCLUDED) == sorted(app_excluded), (
        f"app excludes {sorted(app_excluded)}, this repository excludes {sorted(naming.APP_EXCLUDED)}"
    )


def test_the_vision_exception_matches_the_app():
    # LFM2.5-VL is the one vision family the app will load, checked before the exclusions.
    source = kotlin_source()
    assert '"lfm25vl" in name' in source
    assert naming.app_family("LFM2.5-VL-1.6B") == "lfm25"
    assert naming.app_family("Qwen3.5-VL-2B") is None
