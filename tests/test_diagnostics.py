from __future__ import annotations

from mtplx.diagnostics import (
    DEFAULT_SPEED_MODEL_SIZE_BYTES,
    build_diagnostics_payload,
    required_download_free_bytes,
    write_doctor_bundle,
)


def test_required_download_free_bytes_has_temp_and_headroom() -> None:
    required = required_download_free_bytes(DEFAULT_SPEED_MODEL_SIZE_BYTES)

    assert required > DEFAULT_SPEED_MODEL_SIZE_BYTES
    assert required >= int(DEFAULT_SPEED_MODEL_SIZE_BYTES * 2.5)


def test_diagnostics_payload_has_production_checks(tmp_path) -> None:
    payload = build_diagnostics_payload(
        model_cache=tmp_path,
        mlx_info={"mlx_error": "missing"},
        thermal_control={"available": False},
    )

    assert payload["support_matrix"]["supported"]["default_model"] == (
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
    assert payload["support_matrix"]["supported"]["default_profile"] == "sustained"
    ids = {check["id"] for check in payload["checks"]}
    assert {
        "os.macos_version",
        "python.native_arm64",
        "python.version",
        "mlx.import",
        "resource.memory",
        "resource.model_cache_disk",
        "model.cache",
        "model.default_repo",
        "docker.binary",
        "thermal.control",
        "power.low_power_mode",
        "power.thermal_pressure",
    }.issubset(ids)


def test_default_repo_check_rejects_stale_public_namespace(tmp_path) -> None:
    payload = build_diagnostics_payload(model_cache=tmp_path)
    check = next(item for item in payload["checks"] if item["id"] == "model.default_repo")

    assert check["status"] == "pass"
    assert check["observed"] == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    assert not check["observed"].startswith("mtplx/")


def test_write_doctor_bundle_creates_redacted_zip(tmp_path) -> None:
    report = {
        "environment": {"project_root": "/Users/example/private"},
        "host": {"python_executable": "/Users/example/.venv/bin/python"},
    }

    bundle = write_doctor_bundle(report=report, output_dir=tmp_path)

    assert bundle["redacted"] is True
    assert bundle["bundle_zip"].endswith(".zip")
    assert (tmp_path / bundle["bundle_id"] / "doctor.json").exists()
    assert (tmp_path / f"{bundle['bundle_id']}.zip").exists()
