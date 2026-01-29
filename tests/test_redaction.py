from app import _redact_payload


def test_redact_keys_and_values():
    payload = {
        "api_key": "abcdef0123456789SECRETKEY",
        "user": "bob",
        "nested": {"Authorization": "Bearer ABCDEFGHIJKLMNOPQRST"},
        "notes": "short text",
        "list": ["normal", "longtoken012345678901234567890"]
    }

    red = _redact_payload(payload)
    assert red["api_key"] == "[REDACTED]"
    assert red["nested"]["Authorization"] == "[REDACTED]"
    assert red["user"] == "bob"
    assert red["notes"] == "short text"
    # long token in list should be redacted
    assert any(item == "[REDACTED]" for item in red["list"])