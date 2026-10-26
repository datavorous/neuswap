from model import Intervention
from runtime import Persona, Theatre


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, messages, interventions=(), max_tokens=96):
        self.calls.append((list(messages), tuple(interventions)))
        return {"text": "reply", "seconds": 0.0}


def test_compare_uses_same_prompt_and_switches_without_history():
    engine = FakeEngine()
    registry = {name: Persona(name, name, "exploratory", (Intervention(2, n, 1),))
                for n, name in enumerate(("wizard", "soldier", "jester"))}
    theatre = Theatre(engine, registry)
    theatre.compare("Where is the gate?", experimental=True)
    assert len(engine.calls) == 4
    assert all(call[0] == engine.calls[0][0] for call in engine.calls)
    assert [len(call[1]) for call in engine.calls] == [0, 1, 1, 1]
    assert all(not history for history in theatre.history.values())


def test_histories_are_isolated_and_experimental_is_explicit():
    engine = FakeEngine()
    registry = {name: Persona(name, name, "exploratory", (Intervention(2, 1, 1),))
                for name in ("wizard", "soldier", "jester")}
    theatre = Theatre(engine, registry)
    theatre.ask("wizard", "First question")
    theatre.ask("soldier", "Different question")
    assert engine.calls[0][1] == ()
    assert len(engine.calls[1][0]) == 2
    theatre.ask("wizard", "Second question", experimental=True)
    assert len(engine.calls[2][0]) == 4
    assert len(engine.calls[2][1]) == 1
    assert len(theatre.history["soldier"]) == 2
    theatre.reset("wizard")
    assert theatre.history["wizard"] == []


def test_swap_changes_config_without_reloading_engine():
    engine = FakeEngine()
    registry = {name: Persona(name, name, "unvalidated", (Intervention(2, n, 1),))
                for n, name in enumerate(("wizard", "soldier", "jester"))}
    theatre = Theatre(engine, registry)
    theatre.swap("wizard", "soldier")
    assert theatre.registry["wizard"].interventions[0].neuron == 1
    assert theatre.registry["soldier"].interventions[0].neuron == 0
    assert theatre.registry["wizard"].status == "experimental_swap"
    assert theatre.engine is engine
