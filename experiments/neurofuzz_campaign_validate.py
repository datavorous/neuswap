"""Frozen held-out causal tests for shortlisted campaign candidates.

Write batch1_shortlist.json only after inspecting the selection screen. Each
entry needs id, family, sign, hypothesis, and a novelty_group. This script
then tests exactly that frozen configuration on unrelated prompts.
"""

import json
import random
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neurofuzz_campaign import MANIFEST, ROOT, SYSTEM
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "batch1_shortlist.json"
ROWS = ROOT / "batch1_heldout.jsonl"
PROMPTS = [
    ("bent_photo", "An old family photo has a bent corner. What would you do with it?"),
    ("new_route", "Your usual walking route is blocked by construction. How would you choose another?"),
    ("puzzling_reply", "A classmate says 'interesting' after hearing my idea. How should I interpret that?"),
    ("plant_gift", "Someone gives me a plant without care instructions. What should I do first?"),
    ("queue", "Two people arrive at a service desk at nearly the same time. How should the clerk decide who goes first?"),
    ("unexplained_noise", "I hear a repetitive clicking sound from an appliance. What is a sensible first step?"),
]


def conditions(engine, decoder, candidate, msg):
    rng = random.Random(64071 + sum(ord(x) for x in candidate["id"]))
    family = candidate["family"]
    sign = candidate["sign"]
    if family == "native":
        layer, neuron = candidate["layer"], candidate["neuron"]
        for label, dose in (("half", 0.175), ("full", 0.35), ("opposite", -0.35)):
            calibration = engine.projected_delta_unit(layer, neuron, msg)
            delta = sign * dose * calibration["delta_per_residual"]
            yield label, [Intervention(layer, neuron, delta)], [], {"dose": dose, "delta": delta}
        for i in range(2):
            other = rng.randrange(engine.width)
            while other == neuron:
                other = rng.randrange(engine.width)
            calibration = engine.projected_delta_unit(layer, other, msg)
            delta = sign * 0.35 * calibration["delta_per_residual"]
            yield f"random{i}", [Intervention(layer, other, delta)], [], {
                "dose": 0.35, "delta": delta, "random_neuron": other}
    else:
        fid = candidate["feature_id"]
        vec = decoder[fid]
        target_norm = np.linalg.norm(vec)
        for label, dose in (("half", 2.0), ("full", 4.0), ("opposite", -4.0)):
            shift = sign * dose * vec
            yield label, [], [ResidualShift(18, shift)], {"dose": dose,
                                                           "vector_norm": float(np.linalg.norm(shift))}
        for i in range(2):
            other = rng.randrange(len(decoder))
            while other == fid:
                other = rng.randrange(len(decoder))
            shift = sign * 4.0 * target_norm * decoder[other] / max(np.linalg.norm(decoder[other]), 1e-8)
            yield f"random{i}", [], [ResidualShift(18, shift)], {
                "dose": 4.0, "vector_norm": float(np.linalg.norm(shift)), "random_feature": other}


def main():
    plan = json.loads(PLAN.read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert plan["source_manifest"] == str(MANIFEST)
    assert plan["heldout_prompts"] == [list(p) for p in PROMPTS]
    assert len(plan["candidates"]) <= 4
    known = {(c["family"], c.get("layer"), c.get("neuron"), c.get("feature_id"))
             for c in manifest["candidates"]}
    for c in plan["candidates"]:
        assert (c["family"], c.get("layer"), c.get("neuron"), c.get("feature_id")) in known
        assert c["sign"] in (-1, 1) and c["hypothesis"] and c["novelty_group"]
    seen = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen[(r["candidate"], r["prompt_id"], r["condition"])] = r["answer"]
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    with ROWS.open("a") as out:
        for candidate in plan["candidates"]:
            for prompt_id, prompt in PROMPTS:
                msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
                specs = [("baseline", [], [], {})] + list(conditions(engine, decoder, candidate, msg)) + [("restored_baseline", [], [], {})]
                for name, interventions, shifts, details in specs:
                    key = (candidate["id"], prompt_id, name)
                    if key in seen:
                        continue
                    answer = engine.generate(msg, interventions=interventions,
                                             residual_shifts=shifts, max_tokens=80,
                                             temperature=0)["text"]
                    row = {"candidate": candidate["id"], "prompt_id": prompt_id,
                           "prompt": prompt, "condition": name, "details": details,
                           "answer": answer}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                    seen[key] = answer
                    print(candidate["id"], prompt_id, name, repr(answer[:100]), flush=True)
                assert seen[(candidate["id"], prompt_id, "baseline")] == seen[(candidate["id"], prompt_id, "restored_baseline")]


if __name__ == "__main__":
    main()
