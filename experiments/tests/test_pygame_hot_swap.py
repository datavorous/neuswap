import pytest

from pygame_hot_swap import (BASE_DIALOGUE, CHARACTERS, MODES, PROMPTS, FixedScene,
                             char_delay, compare_same_premise, highlight_spans)


class FakeEngine:
    def __init__(self):
        self.model = object()
        self.calls = []
        self.calibrations = []

    def generate(self, messages, interventions=(), residual_shifts=(), **kwargs):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "shifted": bool(interventions or residual_shifts),
            "interventions": tuple(interventions),
        })
        return {"text": "shifted" if interventions or residual_shifts else "baseline"}

    def projected_delta_unit(self, layer, neuron, messages):
        prompt = messages[-1]["content"]
        self.calibrations.append((layer, neuron, prompt))
        return {"delta_per_residual": 20.0 if prompt == BASE_DIALOGUE else 30.0}


def test_same_premise_uses_one_model_and_restores_baseline():
    engine = FakeEngine()
    original_model = engine.model
    result = compare_same_premise(FixedScene(engine))

    assert result["premise"] == BASE_DIALOGUE
    assert [(row["name"], row["mode"]) for row in result["rows"]] == [
        (name, MODES[name]) for name in CHARACTERS
    ]
    assert all("baseline" not in row for row in result["rows"])
    assert [row["reply"] for row in result["rows"]] == [BASE_DIALOGUE, "shifted", "shifted"]
    assert engine.model is original_model
    assert [call["shifted"] for call in engine.calls] == [False, True, True, False]
    assert all(call["messages"][-1] == {"role": "user", "content": BASE_DIALOGUE}
               for call in engine.calls)
    assert all(call["messages"] == engine.calls[0]["messages"] for call in engine.calls)
    assert "Mochi" not in str(engine.calls[0]["messages"])
    assert "Miso" not in str(engine.calls[0]["messages"])
    assert "Pipkin" not in str(engine.calls[0]["messages"])
    threat = engine.calls[2]["interventions"]
    assert [(item.layer, item.neuron) for item in threat] == [(8, 8437), (8, 8051)]
    assert [item.delta for item in threat] == pytest.approx([6.6, 6.6])


def test_next_pipkin_line_reuses_model_and_recalibrates_threat():
    engine = FakeEngine()
    scene = FixedScene(engine)
    compare_same_premise(scene)
    next_result = compare_same_premise(scene, PROMPTS[1])

    assert next_result["rows"][0]["reply"] == PROMPTS[1]
    assert len(engine.calls) == 8
    assert all(call["messages"] == engine.calls[4]["messages"] for call in engine.calls[4:])
    assert engine.calls[4]["messages"][-1]["content"] == PROMPTS[1]
    assert [item.delta for item in engine.calls[6]["interventions"]] == pytest.approx([9.9, 9.9])
    assert [call[2] for call in engine.calibrations] == [BASE_DIALOGUE] * 2 + [PROMPTS[1]] * 2


def test_unknown_mode_and_typewriter_pacing():
    scene = FixedScene(FakeEngine())
    with pytest.raises(ValueError):
        scene.generate("MOCHI", "unknown")
    assert char_delay(".") > char_delay("a")


def test_highlight_follows_wrapped_phrase_and_finds_repeated_cues():
    text = "Gas supply is being\ncut off; later, gas supply is being cut off."
    spans = highlight_spans(text, ("gas supply is being cut off",))
    assert [text[start:end] for start, end in spans] == [
        "Gas supply is being\ncut off", "gas supply is being cut off"
    ]
