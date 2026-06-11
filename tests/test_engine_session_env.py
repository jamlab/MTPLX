"""Unit tests for engine_session bank-cap env-var overrides."""

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


@pytest.mark.parametrize("raw", ["0", "-1", "0G", "-2G"])
def test_bank_bytes_from_env_nonpositive_uses_default(monkeypatch, raw):
    monkeypatch.setenv("TEST_BANK_BYTES", raw)
    es = _reload_module()
    assert es._bank_bytes_from_env("TEST_BANK_BYTES", 8888) == 8888


def test_short_no_history_api_request_is_foreground_by_default():
    es = _reload_module()
    messages = [
        {"role": "system", "content": "Return only the final answer."},
        {"role": "user", "content": "Compute 17 + 29 + 101."},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={},
            metadata={},
            main_system_hash=None,
        )
        is False
    )


def test_openwebui_task_header_still_marks_background():
    es = _reload_module()
    messages = [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Conversation text"},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={"x-openwebui-task": "title"},
            metadata={},
            main_system_hash=None,
        )
        is True
    )


def test_system_prompt_mismatch_still_marks_background():
    es = _reload_module()
    main_hash = es.hash_text("main chat system")
    messages = [
        {"role": "system", "content": "Return a short title."},
        {"role": "user", "content": "Conversation text"},
    ]

    assert (
        es.is_background_request(
            messages=messages,
            max_tokens=32,
            headers={},
            metadata={},
            main_system_hash=main_hash,
        )
        is True
    )
