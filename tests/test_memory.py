"""Unit tests for AgentMemory and BrainMemory."""

import pytest
from smallclawlm.memory import AgentMemory, BrainMemory


class TestAgentMemory:
    def test_add_and_render(self):
        m = AgentMemory(max_chars=500)
        m.add("First fact")
        rendered = m.render()
        assert "[AGENT MEMORY]" in rendered
        assert "First fact" in rendered
        assert "[END MEMORY]" in rendered

    def test_empty_memory_renders_empty(self):
        m = AgentMemory()
        assert m.render() == ""
        assert m.as_prefix() == ""

    def test_as_prefix(self):
        m = AgentMemory()
        m.add("test fact")
        prefix = m.as_prefix()
        assert prefix.endswith("\n\n")
        assert "test fact" in prefix

    def test_sliding_window_pruning(self):
        m = AgentMemory(max_chars=100, max_facts=20)
        for i in range(20):
            m.add(f"Fact number {i} with some padding text to use chars")
        # Should have pruned to fit in 100 chars
        assert len(m.render()) <= 250  # Allow for header/footer/timestamp overhead
        assert len(m.facts) >= 3  # Always keeps at least 3

    def test_max_facts_pruning(self):
        m = AgentMemory(max_chars=10000, max_facts=5)
        for i in range(10):
            m.add(f"Short fact {i}")
        assert len(m.facts) <= 5

    def test_add_observation_truncates(self):
        m = AgentMemory()
        m.add_observation("deep_research", "x" * 500, max_len=100)
        last = m.facts[-1]
        assert len(last) < 200  # entry + truncation marker

    def test_add_decision(self):
        m = AgentMemory()
        m.add_decision("Need to research fusion", "deep_research(fusion)")
        last = m.facts[-1]
        assert "Decided" in last
        assert "fusion" in last

    def test_clear(self):
        m = AgentMemory()
        m.add("fact 1")
        m.add("fact 2")
        m.clear()
        assert len(m.facts) == 0
        assert m.render() == ""

    def test_summary(self):
        m = AgentMemory()
        m.add("test")
        s = m.summary()
        assert s["fact_count"] == 1
        assert s["latest"] is not None

    def test_repr(self):
        m = AgentMemory(max_chars=500)
        m.add("hi")
        r = repr(m)
        assert "AgentMemory" in r
        assert "1" in r  # 1 fact

    def test_save_and_load(self, tmp_path):
        m = AgentMemory()
        m.add("persistent fact")
        p = tmp_path / "test_memory.json"
        m.save(p)

        m2 = AgentMemory()
        m2.load(p)
        assert len(m2.facts) == 1
        assert "persistent fact" in m2.facts[0]


class TestBrainMemory:
    def test_record_and_get(self):
        bm = BrainMemory()
        bm.record("brain-1", "fusion energy", "15 papers found")
        assert len(bm.get_history("brain-1")) == 1

    def test_render_for_brain(self):
        bm = BrainMemory()
        bm.record("brain-1", "what is fusion?", "hot plasma")
        rendered = bm.render_for_brain("brain-1")
        assert "PREVIOUS QUERIES" in rendered

    def test_empty_brain_renders_empty(self):
        bm = BrainMemory()
        assert bm.render_for_brain("nonexistent") == ""

    def test_pruning(self):
        bm = BrainMemory(max_per_brain=3)
        for i in range(5):
            bm.record("brain-1", f"q{i}", f"a{i}")
        assert len(bm.get_history("brain-1")) == 3

    def test_clear_specific(self):
        bm = BrainMemory()
        bm.record("brain-1", "q1", "a1")
        bm.record("brain-2", "q2", "a2")
        bm.clear("brain-1")
        assert len(bm.get_history("brain-1")) == 0
        assert len(bm.get_history("brain-2")) == 1

    def test_clear_all(self):
        bm = BrainMemory()
        bm.record("brain-1", "q1", "a1")
        bm.record("brain-2", "q2", "a2")
        bm.clear()
        assert len(bm.get_history("brain-1")) == 0
        assert len(bm.get_history("brain-2")) == 0
