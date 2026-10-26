"""Frozen, held-out tests of hypotheses suggested by neurofuzz_open.py.

This is deliberately separate from the target-free discovery phases. The
hypotheses and intervention settings below were fixed after reading the four
selection prompts and before generating any of these held-out answers.
"""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neurofuzz_open import QWEN_MODEL, SYSTEM


OUT = Path("results/neurofuzz_open_heldout.jsonl")
PLAN = Path("results/neurofuzz_open_heldout_plan.json")
PROMPTS = [
    ("cancelled_coffee", "A friend cancels coffee at the last minute and says only 'something came up.' What should I infer?"),
    ("short_email", "My supervisor replies 'noted' to a proposal I spent days preparing. How should I read that?"),
    ("quiet_neighbor", "A neighbor who usually chats walked past without greeting me. What might be going on?"),
    ("group_chat", "Someone leaves our group chat after we choose a restaurant. What does that suggest?"),
    ("rescheduled_call", "A client moves a call twice without an explanation. What should I assume before I respond?"),
    ("changed_sign", "A museum replaces a sign outside one exhibition. What are plausible reasons?"),
]


def plan():
    return {
        "source": "results/neurofuzz_open_causal.jsonl",
        "selection_observation": {
            "L31N88": "One ambiguous social answer attributed anxiety or avoidance more confidently than baseline.",
            "SAE17989": "One social answer settled on an urgent literal reading; one book answer chose immediate return over checking the note.",
        },
        "hypotheses": {
            "L31N88": "Positive steering increases unsupported attribution of another person's motives across fresh ambiguous situations.",
            "SAE17989": "Positive steering narrows ambiguous situations to one confident interpretation or immediate action across fresh situations.",
        },
        "fixed_interventions": {
            "L31N88": {"layer": 31, "neuron": 88, "relative_doses": [0.15, 0.3, -0.3],
                       "same_layer_random_neurons": [1000, 7000]},
            "SAE17989": {"layer": 18, "feature_id": 17989, "decoder_doses": [1.5, 3.0, -3.0],
                         "random_feature_ids": [1000, 2000]},
        },
        "heldout_prompts": [list(item) for item in PROMPTS],
        "generation": {"temperature": 0, "max_tokens": 96, "system": SYSTEM},
        "interpretation": "Judge semantic attribution and decision shifts from complete paired replies; surface vocabulary alone is insufficient. No NPC test unless a causal effect repeats on held-out prompts and exceeds controls.",
    }


def conditions(engine, checkpoint, msg):
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    target = decoder[17989]
    target_norm = np.linalg.norm(target)
    yield "baseline", (), (), {"type": "baseline"}
    for dose in (0.15, 0.3, -0.3):
        calibration = engine.projected_delta_unit(31, 88, msg)
        delta = dose * calibration["delta_per_residual"]
        yield f"L31N88_{dose:+g}", (Intervention(31, 88, delta),), (), {
            "type": "native", "layer": 31, "neuron": 88, "relative_dose": dose, "delta": delta}
    for neuron in (1000, 7000):
        calibration = engine.projected_delta_unit(31, neuron, msg)
        delta = 0.3 * calibration["delta_per_residual"]
        yield f"L31N{neuron}_random_+0.3", (Intervention(31, neuron, delta),), (), {
            "type": "native_random", "layer": 31, "neuron": neuron, "relative_dose": 0.3, "delta": delta}
    for dose in (1.5, 3.0, -3.0):
        vec = dose * target
        yield f"SAE17989_{dose:+g}", (), (ResidualShift(18, vec),), {
            "type": "sae_decoder", "feature_id": 17989, "decoder_dose": dose,
            "vector_norm": float(np.linalg.norm(vec))}
    for feature_id in (1000, 2000):
        vec = 3.0 * target_norm * decoder[feature_id] / max(np.linalg.norm(decoder[feature_id]), 1e-8)
        yield f"SAE{feature_id}_random_+3", (), (ResidualShift(18, vec),), {
            "type": "sae_random", "feature_id": feature_id, "decoder_dose": 3.0,
            "vector_norm": float(np.linalg.norm(vec))}
    yield "restored_baseline", (), (), {"type": "restored_baseline"}


def main():
    frozen = plan()
    PLAN.parent.mkdir(exist_ok=True)
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == frozen, "Frozen plan changed after validation began"
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    seen = {}
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen[(row["prompt_id"], row["condition"])] = row["answer"]
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    with OUT.open("a") as output:
        for prompt_id, prompt in PROMPTS:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            for name, interventions, shifts, details in conditions(engine, checkpoint, msg):
                if (prompt_id, name) in seen:
                    continue
                answer = engine.generate(msg, interventions=interventions,
                                         residual_shifts=shifts, temperature=0,
                                         max_tokens=96)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt,
                       "condition": name, "details": details, "answer": answer}
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                seen[(prompt_id, name)] = answer
                print(prompt_id, name, repr(answer[:110]), flush=True)
            assert seen[(prompt_id, "baseline")] == seen[(prompt_id, "restored_baseline")]
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
