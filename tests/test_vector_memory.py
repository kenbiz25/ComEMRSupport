import os
from core.memory.vector_memory import VectorMemory


def test_add_and_retrieve(tmp_path):
    session_id = "vec-test-1"
    base = str(tmp_path / "vectors")
    vm = VectorMemory(base_dir=base)

    vm.add_entry(session_id, "Patient: fever and cough for 2 days", role="user")
    vm.add_entry(session_id, "Assistant: Asked about breathing difficulties", role="assistant")

    results = vm.get_similar(session_id, "fever and cough", top_k=2)
    assert len(results) >= 1
    assert any("fever" in r["text"].lower() or "cough" in r["text"].lower() for r in results)
