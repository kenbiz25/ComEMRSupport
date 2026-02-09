# tests/test_krio_detection.py
from rag.flow import BotFlow


def test_detect_krio_explicit():
    bf = BotFlow(None, None, None)
    assert bf._detect_krio('krio: wetin yu de do') is True
    assert bf._detect_krio('Please speak Krio') is True


def test_detect_krio_token():
    bf = BotFlow(None, None, None)
    # Tokens like 'wetin' should NOT auto-enable Krio; only explicit initiation enables Krio
    assert bf._detect_krio('wetin na dis?') is False


def test_detect_krio_negative():
    bf = BotFlow(None, None, None)
    assert bf._detect_krio('Hello, how are you?') is False
