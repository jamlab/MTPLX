"""Unit tests for engine_session bank-cap env-var overrides."""
import os
import importlib

import pytest


def _reload_module():
    import mtplx.engine_session
    return importlib.reload(mtplx.engine_session)


def test_bank_bytes_from_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_BANK_BYTES", raising=False)
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 1234) == 1234


def test_bank_bytes_from_env_plain_integer(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "987654321")
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 0) == 987654321


@pytest.mark.parametrize("raw,expected", [
    ("16G", 16 * 1024**3),
    ("16g", 16 * 1024**3),
    ("32G", 32 * 1024**3),
    ("8G",   8 * 1024**3),
    ("512M", 512 * 1024**2),
    ("4K",   4 * 1024),
    ("1T",   1 * 1024**4),
    ("0.5G", int(0.5 * 1024**3)),
])
def test_bank_bytes_from_env_with_suffix(monkeypatch, raw, expected):
    monkeypatch.setenv("TEST_BANK_BYTES", raw)
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 0) == expected


def test_bank_bytes_from_env_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "not-a-number")
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 5555) == 5555


def test_bank_bytes_from_env_empty_string_uses_default(monkeypatch):
    monkeypatch.setenv("TEST_BANK_BYTES", "")
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 7777) == 7777
