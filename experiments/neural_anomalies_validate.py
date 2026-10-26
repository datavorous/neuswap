"""Frozen general-language validation of shortlisted Neural Anomalies effects."""

import json
import random
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neural_anomalies import ROOT, SYSTEM, require_budget
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "shortlist.json"
ROWS = ROOT / "heldout.jsonl"
PROMPTS = [
    ("mistaken_receipt", "A receipt says I bought a coat yesterday, but I remember only trying it on. What might explain that?"),
    ("new_key", "An unfamiliar key appears on my key ring. What should I do before using it?"),
    ("unexpected_apology", "A coworker apologizes for an incident I do not remember. How should I respond?"),
    ("silent_alarm", "The fire alarm flashes but makes no sound. What might be happening?"),
    ("broken_clock", "Two clocks in the same room disagree by an hour. How would you find out which is right?"),
    ("forgotten_promise", "A friend says I promised to help them tomorrow, but I do not recall agreeing. What is a fair response?"),
]


def conditions(engine, decoder, candidate, msg):
    rng = random.Random(90375 + sum(ord(c) for c in candidate["id"]))
    sign = candidate["sign"]
    if candidate["family"] == "native":
        layer, neuron = candidate["layer"], candidate["neuron"]
        full_dose = candidate.get("full_dose", 0.55)
        for label, dose in (("half", full_dose / 2), ("full", full_dose), ("opposite", -full_dose)):
            cal = engine.projected_delta_unit(layer, neuron, msg)
            delta = sign * dose * cal["delta_per_residual"]
            yield label, [Intervention(layer, neuron, delta)], [], {
                "relative_dose": dose, "delta": delta, "neuron": neuron}
        used = {neuron}
        for i in range(2):
            other = rng.randrange(engine.width)
            while other in used:
                other = rng.randrange(engine.width)
            used.add(other)
            cal = engine.projected_delta_unit(layer, other, msg)
            delta = sign * full_dose * cal["delta_per_residual"]
            yield f"random{i}", [Intervention(layer, other, delta)], [], {
                "relative_dose": full_dose, "delta": delta, "random_neuron": other}
    else:
        fid = candidate["feature_id"]
        target = decoder[fid]
        norm = np.linalg.norm(target)
        full_dose = candidate.get("full_dose", 6.0)
        for label, dose in (("half", full_dose / 2), ("full", full_dose), ("opposite", -full_dose)):
            vec = sign * dose * target
            yield label, [], [ResidualShift(18, vec)], {
                "decoder_dose": dose, "vector_norm": float(np.linalg.norm(vec))}
        used = {fid}
        for i in range(2):
            other = rng.randrange(len(decoder))
            while other in used:
                other = rng.randrange(len(decoder))
            used.add(other)
            vec = sign * full_dose * norm * decoder[other] / max(np.linalg.norm(decoder[other]), 1e-8)
            yield f"random{i}", [], [ResidualShift(18, vec)], {
                "decoder_dose": full_dose, "vector_norm": float(np.linalg.norm(vec)),
                "random_feature_id": other}


def main():
    frozen = json.loads(PLAN.read_text())
    manifest = json.loads(Path(frozen["source_manifest"]).read_text())
    assert frozen["prompts"] == [list(x) for x in PROMPTS]
    assert 1 <= len(frozen["candidates"]) <= 4
    known = {(c["family"], c.get("layer"), c.get("neuron"), c.get("feature_id"))
             for c in manifest["candidates"]}
    for c in frozen["candidates"]:
        assert (c["family"], c.get("layer"), c.get("neuron"), c.get("feature_id")) in known
        assert c["sign"] in (-1, 1) and c["hypothesis"] and c["novelty_group"]
    seen = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen[(r["candidate"], r["prompt_id"], r["condition"])] = r["answer"]
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    with ROWS.open("a") as out:
        for candidate in frozen["candidates"]:
            for prompt_id, prompt in PROMPTS:
                msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
                specs = [("baseline", [], [], {})] + list(conditions(engine, decoder, candidate, msg)) + [("restored_baseline", [], [], {})]
                for name, interventions, shifts, details in specs:
                    key = (candidate["id"], prompt_id, name)
                    if key in seen:
                        continue
                    require_budget()
                    answer = engine.generate(msg, interventions=interventions,
                                             residual_shifts=shifts, max_tokens=96,
                                             temperature=0)["text"]
                    row = {"candidate": candidate["id"], "prompt_id": prompt_id,
                           "prompt": prompt, "condition": name, "details": details,
                           "answer": answer}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                    seen[key] = answer
                    print(candidate["id"], prompt_id, name, repr(answer[:110]), flush=True)
                assert seen[(candidate["id"], prompt_id, "baseline")] == seen[(candidate["id"], prompt_id, "restored_baseline")]


if __name__ == "__main__":
    main()
