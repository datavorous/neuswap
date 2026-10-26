import mlx.core as mx
import numpy as np
import pytest

from model import (Intervention, OutputShift, ResidualPatch, ResidualShift,
                   TheatreModel, TokenNeuronEdit, TokenNeuronOffset, TokenResidualEdit)


@pytest.fixture(scope="module")
def engine():
    return TheatreModel.from_cache()


MESSAGES = [{"role": "user", "content": "The bridge is closed. What should we do?"}]


def test_zero_switch_and_restoration_without_reload(engine):
    model_id = id(engine.model)
    weight = engine.model.model.layers[3].mlp.base.down_proj.weight
    before_weight = mx.array(weight)
    mx.eval(before_weight)
    baseline = engine.generate(MESSAGES, max_tokens=24)["text"]
    zero = engine.generate(MESSAGES, [Intervention(3, 100, 0)], max_tokens=24)["text"]
    engine.generate(MESSAGES, [Intervention(3, 100, 2)], max_tokens=24)
    restored = engine.generate(MESSAGES, max_tokens=24)["text"]
    assert baseline == zero == restored
    assert id(engine.model) == model_id
    assert bool(mx.all(weight == before_weight))
    assert not engine._active


def test_only_intended_intermediate_coordinate_changes(engine):
    before, after = engine.activation_pair(MESSAGES, "Take the river path.", 3,
                                            [Intervention(3, 100, 2)])
    difference = after - before
    assert abs(float(mx.mean(difference[:, 100])) - 2) < 0.01
    others = mx.concatenate([difference[:, :100], difference[:, 101:]], axis=-1)
    assert float(mx.max(mx.abs(others))) == 0.0


def test_exception_clears_state(engine):
    with pytest.raises(RuntimeError):
        with engine.using([Intervention(2, 25, -1)]):
            assert engine._active
            raise RuntimeError("simulated generation failure")
    assert not engine._active
    assert engine.capture_layer is None


def test_projected_calibration_restores_layer(engine):
    layer = engine.model.model.layers[3]
    calibration = engine.projected_delta_unit(3, 100, MESSAGES)
    assert calibration["residual_norm"] > 0
    assert calibration["column_norm"] > 0
    assert engine.model.model.layers[3] is layer


def test_output_direction_changes_only_requested_residual_coordinate(engine):
    mlp = engine.model.model.layers[3].mlp
    x = mx.zeros((1, 1, engine.residual_width))
    baseline = mlp(x)
    vector = np.zeros(engine.residual_width, dtype=np.float32)
    vector[17] = 2.0
    with engine.using(output_shifts=[OutputShift(3, vector)]):
        steered = mlp(x)
        mx.eval(baseline, steered)
    difference = steered - baseline
    assert float(difference[0, 0, 17]) == 2.0
    assert float(mx.sum(mx.abs(difference))) == 2.0
    assert not engine._output_active


def test_residual_edit_restores_layer_and_zero_shift(engine):
    layer = engine.model.model.layers[3]
    baseline = engine.generate(MESSAGES, max_tokens=24)["text"]
    zero = engine.generate(MESSAGES, max_tokens=24,
                           residual_shifts=[ResidualShift(3, np.zeros(engine.residual_width))])["text"]
    assert zero == baseline
    assert engine.model.model.layers[3] is layer
    with pytest.raises(RuntimeError):
        with engine.using(residual_patches=[ResidualPatch(3, np.zeros(engine.residual_width))]):
            assert engine.model.model.layers[3] is not layer
            raise RuntimeError("simulated patch failure")
    assert engine.model.model.layers[3] is layer
    assert not engine._residual_active and not engine._residual_patches


def test_final_prompt_residual_patch_is_applied_once(engine):
    baseline = engine.generate(MESSAGES, max_tokens=12)["text"]
    patched = engine.generate(
        MESSAGES, max_tokens=12,
        residual_patches=[ResidualPatch(3, np.zeros(engine.residual_width))])
    restored = engine.generate(MESSAGES, max_tokens=12)["text"]
    assert patched["patch_applied_layers"] == [3]
    assert engine._generation_patch_pending is None
    assert baseline == restored


def test_token_local_noop_and_cleanup(engine):
    messages = [{"role":"user", "content":"What does the quoted word 'pajeet' mean?"}]
    ids = engine.tokenizer.encode(engine.prompt(messages), add_special_tokens=False)
    position = ids.index(engine.tokenizer.encode("pajeet", add_special_tokens=False)[-1])
    original = engine.model.model.layers[3]
    baseline = engine.generate(messages, max_tokens=16)["text"]
    unchanged = engine.generate(
        messages, max_tokens=16,
        token_residual_edits=[TokenResidualEdit(3, (position,), np.zeros(engine.residual_width))],
        token_neuron_edits=[TokenNeuronEdit(3, 100, (position,), 1.0)],
        token_neuron_offsets=[TokenNeuronOffset(3, 101, (position,), 0.0)])["text"]
    assert unchanged == baseline
    assert engine.model.model.layers[3] is original
    assert not engine._token_residual_edits and not engine._token_neuron_edits
    assert not engine._token_neuron_offsets
    with pytest.raises(RuntimeError):
        with engine.using(token_residual_edits=[TokenResidualEdit(3, (position,),
                                         np.ones(engine.residual_width))]):
            raise RuntimeError("simulated token edit failure")
    assert engine.model.model.layers[3] is original
    assert not engine._token_residual_edits


def test_token_neuron_offset_touches_one_coordinate_and_position(engine):
    ids = engine.tokenizer.encode(engine.prompt(MESSAGES), add_special_tokens=False)
    with engine.using(token_neuron_offsets=[TokenNeuronOffset(3, 100, (5,), 2.0)]):
        engine.capture_layer = 3
        engine.model(mx.array([ids]))
        difference = engine.captured_after - engine.captured_before
        mx.eval(difference)
        expected = mx.zeros_like(difference).at[:, 5, 100].add(2.0)
        assert bool(mx.all(mx.abs(difference - expected) < 0.01))
    assert not engine._token_neuron_offsets
