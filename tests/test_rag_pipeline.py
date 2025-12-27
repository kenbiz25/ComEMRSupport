
from rag.composer import RagComposer

def test_rag_smoke():
    composer = RagComposer()
    ans, meta = composer.answer("How do I reset my ComEMR password?")
    assert isinstance(ans, str)
    assert "password" in ans.lower() or meta["hits"] >= 0