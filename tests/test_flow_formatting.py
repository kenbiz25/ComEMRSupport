# tests/test_flow_formatting.py
from rag.flow import BotFlow


def test_remove_asterisks_and_clean():
    bf = BotFlow(None, None, None)
    text = "Hello! *Please* follow these steps:\n1. Do this\n2. Do that"
    out = bf._format_outgoing(text)
    assert "*" not in out
    assert "1. Do this" in out


def test_limit_list_to_five_items():
    bf = BotFlow(None, None, None)
    text = "Steps:\n1. a\n2. b\n3. c\n4. d\n5. e\n6. f\n7. g"
    out = bf._format_outgoing(text)
    assert "6." not in out
    assert "..." in out


def test_avoid_lone_item_cutoff():
    bf = BotFlow(None, None, None)
    text = "1. One\n2. Two\n3. Three\n4. Four\n5."  # last item incomplete
    out = bf._format_outgoing(text)
    # last lonely empty item should be ignored, no dangling "5." left
    assert "5." not in out or "5. " in out and out.strip().endswith('...') or out.strip().endswith('.')
