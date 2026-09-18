"""Publishing one window into a repo that already holds other windows and backends."""

import json
from types import SimpleNamespace

import pytest

from pipeline import manifest, publish


def report(context, backend="xnnpack", target=None, sha="abcdef0123456789", run="https://gh/run/1"):
    label = f"{context // 1024}k"
    folder = backend if target is None else f"{backend}/{target}"
    return {
        "backend": backend,
        "target": target,
        "tokenizer": "tokenizer.json",
        "output_repo": "experimentalmachines/Qwen3-1.7B-ExecuTorch",
        "source": {"id": "Qwen/Qwen3-1.7B", "sha": sha, "license": {"license": "apache-2.0"}, "license_files": []},
        "toolchain": {"executorch": "1.4.0"},
        "recipe": {"label": "8da4w-g32, int8 embeddings", "description": "recipe text."},
        "window": {"context": context, "kv_cache_bytes_per_token": 229_376, "fits_phone_budget": context <= 8192},
        "files": [{"path": f"{folder}/Qwen3-1.7B-8da4w-{label}.pte", "bytes": 1_000 * context, "sha256": "ff"}],
        "metadata": {"get_max_context_len": context, "get_max_seq_len": min(2048, context)},
        "smoke": {"passed": True, "answered": True},
        "run": {"url": run},
    }


def test_superseded_only_touches_this_window_and_the_pre_matrix_files():
    folder, label = "xnnpack", "2k"
    keep = [
        "xnnpack/Qwen3-1.7B-8da4w-32k.pte",
        "xnnpack/export-report-32k.json",
        "qnn/sm8650/Qwen3-1.7B-qnn-hybrid-2k.pte",
    ]
    replace = [
        "xnnpack/Qwen3-1.7B-8da4w-2k.pte",
        "xnnpack/export-report-2k.json",
        "xnnpack/config.json",
        "xnnpack/export-report.json",
    ]
    assert [p for p in keep if publish.superseded(p, folder, label)] == []
    assert [p for p in replace if publish.superseded(p, folder, label)] == replace
    # "2k" is not a substring match on "32k", and a chip folder is not its backend's folder.
    assert not publish.superseded("xnnpack/Qwen3-1.7B-8da4w-32k.pte", "xnnpack", "2k")
    assert not publish.superseded("qnn/sm8650/config.json", "qnn", "2k")
    assert publish.superseded("mtk/mt6991/Qwen3-1.7B-neuropilot-a16w4-2k-chunk1of4.pte", "mtk/mt6991", "2k")
    assert not publish.superseded("mtk/mt6991/Qwen3-1.7B-neuropilot-embedding-fp32.bin", "mtk/mt6991", "2k")
    # A model whose own name carries a window-like token (a real HuggingFaceTB repo).
    name = "xnnpack/SmolLM2-1.7B-Instruct-16k-8da4w-4k.pte"
    assert publish.window_of(name.split("/")[1]) == "4k"
    assert not publish.superseded(name, "xnnpack", "16k")
    assert publish.superseded(name, "xnnpack", "4k")
    assert publish.window_of("export-report-16k.json") == "16k"
    assert publish.window_of("config.json") is None


def test_release_tags_are_per_window_and_chip():
    assert publish.release_tag(report(4096)) == "Qwen3-1.7B-xnnpack-4k-abcdef0"
    assert publish.release_tag(report(2048, backend="qnn", target="sm8650")) == "Qwen3-1.7B-qnn-sm8650-2k-abcdef0"


def test_publish_hf_retries_once_on_a_stale_parent_commit(tmp_path, monkeypatch):
    from huggingface_hub.errors import HfHubHTTPError

    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    hub = FakeHub({"README.md": None})
    calls = []

    def create_commit(repo_id, operations, commit_message, parent_commit):
        calls.append(parent_commit)
        if len(calls) == 1:  # another window's job committed in between
            import httpx

            response = httpx.Response(412, request=httpx.Request("POST", "https://huggingface.co/api"))
            raise HfHubHTTPError("412 Precondition Failed", response=response)
        return SimpleNamespace(commit_url="https://hf/commit/2")

    hub.create_commit = create_commit
    monkeypatch.setattr(publish.hub, "api", lambda: hub)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.publish_hf(out, "xnnpack") == "https://hf/commit/2"
    assert calls == ["rev1", "rev1"]  # re-read the repo and rebuilt the commit before retrying


def test_a_concurrent_commit_is_retried_too(tmp_path, monkeypatch):
    """409, not just 412. Five windows publish at once and the Hub answers either way.

    Run 35373841386 published four of five windows and lost 8192 to a 409 that was not
    retried, so the repository carried four of the five files it should have.
    """
    from huggingface_hub.errors import HfHubHTTPError

    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    hub = FakeHub({"README.md": None})
    calls = []

    def create_commit(repo_id, operations, commit_message, parent_commit):
        calls.append(parent_commit)
        if len(calls) == 1:
            import httpx

            response = httpx.Response(409, request=httpx.Request("POST", "https://huggingface.co/api"))
            raise HfHubHTTPError("409 Conflict", response=response)
        return SimpleNamespace(commit_url="https://hf/commit/2")

    hub.create_commit = create_commit
    monkeypatch.setattr(publish.hub, "api", lambda: hub)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    assert publish.publish_hf(out, "xnnpack") == "https://hf/commit/2"
    assert calls == ["rev1", "rev1"]


def test_a_status_that_is_not_a_race_still_raises(tmp_path, monkeypatch):
    # Retrying a 403 or a 404 would turn a real failure into six slow ones and then the
    # same failure, with the cause five minutes further from the top of the log.
    from huggingface_hub.errors import HfHubHTTPError

    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-2k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-2k.json").write_text(json.dumps(report(2048)))
    hub = FakeHub({"README.md": None})
    calls = []

    def create_commit(repo_id, operations, commit_message, parent_commit):
        import httpx

        calls.append(parent_commit)
        response = httpx.Response(403, request=httpx.Request("POST", "https://huggingface.co/api"))
        raise HfHubHTTPError("403 Forbidden", response=response)

    hub.create_commit = create_commit
    monkeypatch.setattr(publish.hub, "api", lambda: hub)
    monkeypatch.setattr(publish.time, "sleep", lambda s: None)
    with pytest.raises(HfHubHTTPError):
        publish.publish_hf(out, "xnnpack")
    assert len(calls) == 1


def test_backend_config_merges_every_window_in_the_folder():
    config = manifest.backend_config([report(8192), report(2048), report(32768)])
    assert [v["context"] for v in config["variants"]] == [2048, 8192, 32768]
    assert [v["fits_phone_budget"] for v in config["variants"]] == [True, True, False]
    assert config["variants"][0]["file"] == "Qwen3-1.7B-8da4w-2k.pte"
    assert config["variants"][0]["methods"]["get_max_context_len"] == 2048


class FakeHub:
    """Enough of HfApi for publish_hf: a repo with files, and a create_commit that records."""

    def __init__(self, files: dict[str, dict]):
        self.files = files  # path -> report json for report files
        self.token = None
        self.commits = []

    def create_repo(self, repo_id, repo_type, exist_ok):
        pass

    def model_info(self, repo_id, expand):
        return SimpleNamespace(sha="rev1", siblings=[SimpleNamespace(rfilename=p) for p in self.files])

    def create_commit(self, repo_id, operations, commit_message, parent_commit):
        self.commits.append((operations, commit_message, parent_commit))
        return SimpleNamespace(commit_url="https://hf/commit/1")


def test_publish_hf_keeps_the_other_windows_and_regenerates_config_and_readme(tmp_path, monkeypatch):
    out = tmp_path / "out"
    (out / "xnnpack").mkdir(parents=True)
    (out / "tokenizer.json").write_text("{}")
    new = report(4096)
    (out / "xnnpack" / "Qwen3-1.7B-8da4w-4k.pte").write_bytes(b"pte")
    (out / "xnnpack" / "export-report-4k.json").write_text(json.dumps(new))
    (out / "xnnpack" / "config.json").write_text(json.dumps(manifest.backend_config(new)))

    remote = {
        "README.md": None,
        "tokenizer.json": None,
        "xnnpack/Qwen3-1.7B-8da4w-16k.pte": None,
        "xnnpack/export-report-16k.json": report(16384, run="https://gh/run/0"),
        "xnnpack/config.json": None,
        # A pre-matrix export: unlabelled report, its window's file, an older 4k that is replaced.
        "xnnpack/export-report.json": report(8192),
        "xnnpack/Qwen3-1.7B-8da4w-8k.pte": None,
        "xnnpack/Qwen3-1.7B-8da4w-4k.pte": None,
        "qnn/sm8650/Qwen3-1.7B-qnn-hybrid-2k.pte": None,
        "qnn/sm8650/export-report-2k.json": report(2048, backend="qnn", target="sm8650"),
        "qnn/sm8650/config.json": None,
    }
    hub = FakeHub(remote)
    monkeypatch.setattr(publish.hub, "api", lambda: hub)

    def fake_download(repo_id, path, revision, token):
        local = tmp_path / "dl" / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(remote[path]))
        return str(local)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)

    assert publish.publish_hf(out, "xnnpack") == "https://hf/commit/1"
    [(operations, message, parent)] = hub.commits
    assert parent == "rev1" and message.startswith("xnnpack 4k: Qwen/Qwen3-1.7B@")
    deleted = sorted(op.path_in_repo for op in operations if type(op).__name__ == "CommitOperationDelete")
    added = {op.path_in_repo: op for op in operations if type(op).__name__ == "CommitOperationAdd"}
    # Only the pre-matrix report goes, moved to its labelled name; the 8k and 16k windows and
    # the QNN folder stay.
    assert deleted == ["xnnpack/export-report.json"]
    assert json.loads(added["xnnpack/export-report-8k.json"].path_or_fileobj)["window"]["context"] == 8192
    assert "xnnpack/Qwen3-1.7B-8da4w-4k.pte" in added and "tokenizer.json" in added
    config = json.loads(added["xnnpack/config.json"].path_or_fileobj)
    assert [v["context"] for v in config["variants"]] == [4096, 8192, 16384]
    readme = added["README.md"].path_or_fileobj.decode()
    for name in (
        "xnnpack/Qwen3-1.7B-8da4w-4k.pte",
        "8da4w-8k.pte",
        "8da4w-16k.pte",
        "qnn/sm8650/Qwen3-1.7B-8da4w-2k.pte",
    ):
        assert name in readme, name
    assert "[run 1](https://gh/run/0)" in readme and "[run 2](https://gh/run/1)" in readme


def test_load_report_wants_exactly_one_report(tmp_path):
    folder = tmp_path / "xnnpack"
    folder.mkdir()
    with pytest.raises(ValueError, match="expected one export report"):
        publish.load_report(tmp_path, "xnnpack")
    (folder / "export-report-2k.json").write_text(json.dumps(report(2048)))
    assert publish.load_report(tmp_path, "xnnpack")["window"]["context"] == 2048
    # Several windows exported into one folder by hand: --context picks.
    (folder / "export-report-8k.json").write_text(json.dumps(report(8192)))
    with pytest.raises(ValueError, match="pass --context"):
        publish.load_report(tmp_path, "xnnpack")
    assert publish.load_report(tmp_path, "xnnpack", context=8192)["window"]["context"] == 8192
