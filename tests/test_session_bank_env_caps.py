"""Unit tests for SessionBank cap env-var overrides wired in engine_session.

Covers the entry-count override (MTPLX_SESSION_BANK_MAX_ENTRIES) added on top
of the existing byte-cap envs (MTPLX_SESSION_BANK_MAX_BYTES,
MTPLX_SESSION_BANK_PER_SESSION_BYTES). The byte-cap helper is exercised in
detail in tests/test_engine_session_env.py; the regression cases here just
confirm all three envs compose correctly when wired through
EngineSessionManager.__init__.
"""
from __future__ import annotations

import importlib
import logging

import pytest


def _reload_engine_session():
    import mtplx.engine_session
    return importlib.reload(mtplx.engine_session)


# --- _bank_entries_from_env helper ----------------------------------------


def test_bank_entries_from_env_default_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_BANK_ENTRIES", raising=False)
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 8


def test_bank_entries_from_env_default_when_empty(monkeypatch):
    monkeypatch.setenv("TEST_BANK_ENTRIES", "")
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 8


def test_bank_entries_from_env_valid_integer(monkeypatch):
    monkeypatch.setenv("TEST_BANK_ENTRIES", "24")
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 24


def test_bank_entries_from_env_strips_whitespace(monkeypatch):
    monkeypatch.setenv("TEST_BANK_ENTRIES", "  16  ")
    es = _reload_engine_session()
    assert es._bank_entries_from_env("TEST_BANK_ENTRIES", 8) == 16


@pytest.mark.parametrize("raw", ["abc", "12.5", "1e2", "0x10", ""])
def test_bank_entries_from_env_invalid_falls_back(monkeypatch, caplog, raw):
    monkeypatch.setenv("TEST_BANK_ENTRIES", raw)
    es = _reload_engine_session()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mtplx.engine_session"):
        result = es._bank_entries_from_env("TEST_BANK_ENTRIES", 8)
    assert result == 8
    if raw == "":
        # Empty string is the unset-equivalent path; no warning expected.
        assert not any(
            "TEST_BANK_ENTRIES" in rec.getMessage() for rec in caplog.records
        )
    else:
        assert any(
            "TEST_BANK_ENTRIES" in rec.getMessage()
            and rec.levelno == logging.WARNING
            for rec in caplog.records
        ), f"expected warning for raw={raw!r}, got {caplog.records!r}"


@pytest.mark.parametrize("raw", ["0", "-1", "-3"])
def test_bank_entries_from_env_below_one_falls_back(monkeypatch, caplog, raw):
    monkeypatch.setenv("TEST_BANK_ENTRIES", raw)
    es = _reload_engine_session()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mtplx.engine_session"):
        result = es._bank_entries_from_env("TEST_BANK_ENTRIES", 8)
    assert result == 8
    assert any(
        "TEST_BANK_ENTRIES" in rec.getMessage()
        and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"expected warning for raw={raw!r}, got {caplog.records!r}"


# --- EngineSessionManager wiring ------------------------------------------


def test_manager_uses_env_max_entries(monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", "24")
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 24


def test_manager_default_max_entries_when_unset(monkeypatch):
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()
    # Mirrors SessionBank's public default.
    assert mgr.bank.max_entries == 8


@pytest.mark.parametrize("raw", ["abc", "0", "-3"])
def test_manager_invalid_max_entries_falls_back_to_default(
    monkeypatch, caplog, raw
):
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raw)
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_BYTES", raising=False)
    monkeypatch.delenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", raising=False)
    es = _reload_engine_session()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="mtplx.engine_session"):
        mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 8
    assert any(
        "MTPLX_SESSION_BANK_MAX_ENTRIES" in rec.getMessage()
        and rec.levelno == logging.WARNING
        for rec in caplog.records
    )


def test_manager_byte_caps_still_work_alongside_entries(monkeypatch):
    """Regression: setting all three env vars composes correctly."""
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_ENTRIES", "20")
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "32G")
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "16G")
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 20
    assert mgr.bank.max_bytes == 32 * 1024**3
    assert mgr.bank.per_session_max_bytes == 16 * 1024**3


def test_manager_byte_caps_alone_still_work(monkeypatch):
    """Regression: byte-cap envs without the new entries env still work
    and leave max_entries at the default."""
    monkeypatch.delenv("MTPLX_SESSION_BANK_MAX_ENTRIES", raising=False)
    monkeypatch.setenv("MTPLX_SESSION_BANK_MAX_BYTES", "16G")
    monkeypatch.setenv("MTPLX_SESSION_BANK_PER_SESSION_BYTES", "8G")
    es = _reload_engine_session()
    mgr = es.EngineSessionManager()
    assert mgr.bank.max_entries == 8
    assert mgr.bank.max_bytes == 16 * 1024**3
    assert mgr.bank.per_session_max_bytes == 8 * 1024**3
