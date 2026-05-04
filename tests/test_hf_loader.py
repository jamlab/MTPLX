from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

from mtplx.hf_loader import (
    cached_model_is_complete,
    cached_model_path,
    hf_cache_report,
    list_cached_models,
    pull_model,
    remove_cached_model,
    repo_id_from_model_ref,
    resolve_model_path,
    safe_model_name,
    validate_mtplx_model_files,
)


def test_repo_id_from_model_ref_accepts_hf_url_and_repo_id():
    assert repo_id_from_model_ref("mtplx/example") == "mtplx/example"
    assert (
        repo_id_from_model_ref("https://huggingface.co/mtplx/example/tree/main")
        == "mtplx/example"
    )
    assert repo_id_from_model_ref("models/local-model") is None


def test_safe_model_name_and_cache_path(tmp_path: Path):
    assert safe_model_name("mtplx/example") == "mtplx--example"
    assert cached_model_path("mtplx/example", cache_dir=tmp_path) == tmp_path / "mtplx--example"


def test_resolve_model_path_uses_cache_for_hf_refs(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors").write_bytes(b"1234")

    assert resolve_model_path("mtplx/example", cache_dir=tmp_path) == cached


def test_cached_model_is_complete_rejects_interrupted_indexed_download(tmp_path: Path):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )

    assert cached_model_is_complete(cached) is False


def test_pull_model_reuses_complete_destination_without_redownload(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    (cached / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    def fail_snapshot_download(**_kwargs):
        raise AssertionError("complete cached model should not download again")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fail_snapshot_download),
    )

    result = pull_model("mtplx/example", cache_dir=tmp_path)

    assert result["path"] == str(cached)
    assert result["reused_existing"] is True
    assert result["resumed_existing"] is False


def test_pull_model_resumes_incomplete_destination(
    tmp_path: Path, monkeypatch
):
    cached = tmp_path / "mtplx--example"
    cached.mkdir()
    (cached / "config.json").write_text("{}\n", encoding="utf-8")
    (cached / "model.safetensors.index.json").write_text(
        '{"weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )
    download_cache = cached / ".cache" / "huggingface" / "download"
    download_cache.mkdir(parents=True)
    (download_cache / "model-00001-of-00001.safetensors.incomplete").write_bytes(
        b"partial"
    )

    def fake_snapshot_download(**kwargs):
        destination = Path(kwargs["local_dir"])
        (destination / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        return str(destination)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    events: list[dict] = []

    result = pull_model(
        "mtplx/example",
        cache_dir=tmp_path,
        progress_callback=events.append,
        progress_interval_s=0,
    )

    assert result["path"] == str(cached)
    assert result["reused_existing"] is False
    assert result["resumed_existing"] is True
    assert result["started_size_bytes"] > 0
    assert [event["event"] for event in events] == ["resume", "complete"]


def test_resolve_model_path_reports_missing_cache(tmp_path: Path):
    try:
        resolve_model_path("mtplx/example", cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert "mtplx pull mtplx/example" in str(exc)
    else:
        raise AssertionError("expected missing cache error")


def test_list_and_remove_cached_models(tmp_path: Path):
    (tmp_path / ".tmp").mkdir()
    model = tmp_path / "mtplx--example"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text("{}\n", encoding="utf-8")
    (model / "small.bin").write_bytes(b"1234")

    rows = list_cached_models(cache_dir=tmp_path)

    assert len(rows) == 1
    assert rows[0].repo_id == "mtplx/example"
    assert rows[0].has_config is True
    assert rows[0].has_runtime_contract is True
    assert rows[0].validation["missing_files"]
    assert rows[0].to_dict()["recommended_profile"] is None
    assert rows[0].size_bytes >= 4

    removed = remove_cached_model("mtplx/example", cache_dir=tmp_path)
    assert removed["removed"] is True
    assert not model.exists()


def test_hf_cache_report_is_no_network(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    cache = tmp_path / "missing-cache"
    report = hf_cache_report(cache_dir=cache)

    assert report["cache_dir"] == str(cache)
    assert report["cache_exists"] is False
    assert report["cached_models"] == 0
    assert "token_present" in report
    assert "disk_free_bytes" in report


def test_validate_mtplx_model_files_reports_required_payload(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    for name in (
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "mtp.safetensors",
    ):
        (model / name).write_text("{}\n", encoding="utf-8")
    (model / "mtplx_runtime.json").write_text('{"arch_id": "qwen3-next-mtp"}\n', encoding="utf-8")

    validation = validate_mtplx_model_files(model)

    assert validation["ok"] is True
    assert validation["missing_files"] == []
    assert validation["contract_arch_id"] == "qwen3-next-mtp"
