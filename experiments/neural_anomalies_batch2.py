"""Second bounded random-component screen for the separate anomalies campaign."""

import json
import random

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neural_anomalies import MANIFEST, ROOT, SYSTEM, previous_components, require_budget
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "batch2_plan.json"
ROWS = ROOT / "screen_batch2.jsonl"
PROMPTS = [
    ("bank_mismatch", "The bank balance is lower than yesterday, but the transaction list shows no new charges. What should I check?"),
    ("salt_water", "Why does a spoonful of salt disappear when stirred into water?"),
    ("odd_greeting", "A longtime coworker greets me as if we have never met. How should I respond?"),
    ("journal_line", "A line appears in my notebook in familiar handwriting, but I cannot remember writing it. What might explain that?"),
]


def make_plan(engine, sae_width):
    prior = json.loads(MANIFEST.read_text())["candidates"]
    native_excluded, sae_excluded = previous_components()
    native_excluded.update((c["layer"], c["neuron"]) for c in prior if c["family"] == "native")
    sae_excluded.update(c["feature_id"] for c in prior if c["family"] == "sae")
    rng = random.Random(198305)
    candidates = []
    while sum(c["family"] == "native" for c in candidates) < 12:
        layer, neuron = rng.randrange(8, 32), rng.randrange(engine.width)
        if (layer, neuron) in native_excluded:
            continue
        native_excluded.add((layer, neuron))
        candidates.append({"family": "native", "layer": layer, "neuron": neuron,
                           "id": f"L{layer}N{neuron}"})
    while sum(c["family"] == "sae" for c in candidates) < 12:
        fid = rng.randrange(sae_width)
        if fid in sae_excluded:
            continue
        sae_excluded.add(fid)
        candidates.append({"family": "sae", "feature_id": fid, "id": f"SAE{fid}"})
    return {"model": QWEN_MODEL, "prompts": PROMPTS, "candidates": candidates,
            "native_relative_dose": 0.7, "sae_decoder_dose": 7.5,
            "max_tokens": 88, "temperature": 0, "seed": 198305}


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    frozen = make_plan(engine, len(decoder))
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == frozen
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    seen = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen[(r["prompt_id"], r["condition"])] = r["answer"]
    with ROWS.open("a") as out:
        for prompt_id, prompt in PROMPTS:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            specs = [("baseline", None, 0)] + [(c["id"] + f"_{sign:+}", c, sign)
                                                 for c in frozen["candidates"] for sign in (-1, 1)]
            specs.append(("restored_baseline", None, 0))
            for name, candidate, sign in specs:
                if (prompt_id, name) in seen:
                    continue
                interventions, shifts, details = [], [], {}
                if candidate and candidate["family"] == "native":
                    layer, neuron = candidate["layer"], candidate["neuron"]
                    cal = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * frozen["native_relative_dose"] * cal["delta_per_residual"]
                    interventions = [Intervention(layer, neuron, delta)]
                    details = {"layer": layer, "neuron": neuron, "sign": sign,
                               "relative_dose": frozen["native_relative_dose"], "delta": delta}
                elif candidate:
                    fid = candidate["feature_id"]
                    vec = sign * frozen["sae_decoder_dose"] * decoder[fid]
                    shifts = [ResidualShift(18, vec)]
                    details = {"feature_id": fid, "sign": sign,
                               "decoder_dose": frozen["sae_decoder_dose"],
                               "vector_norm": float(np.linalg.norm(vec))}
                require_budget()
                answer = engine.generate(msg, interventions=interventions,
                                         residual_shifts=shifts, max_tokens=88,
                                         temperature=0)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt,
                       "condition": name, "details": details, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                seen[(prompt_id, name)] = answer
                print(prompt_id, name, repr(answer[:90]), flush=True)
            assert seen[(prompt_id, "baseline")] == seen[(prompt_id, "restored_baseline")]


if __name__ == "__main__":
    main()
