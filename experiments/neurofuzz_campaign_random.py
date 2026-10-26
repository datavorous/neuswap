"""Third bounded screen: fresh random coordinates after the ranked and combo batches."""

import json
import random

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neurofuzz_campaign import MANIFEST, ROOT, SYSTEM, excluded_native
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "batch3_random_plan.json"
ROWS = ROOT / "batch3_random_screen.jsonl"
PROMPTS = [
    ("group_decision", "Our team can either fix an old tool or build a replacement. How should we decide?"),
    ("misplaced_note", "A handwritten note appears on my desk with no name. How might I interpret it?"),
]


def plan(engine, checkpoint):
    previous = json.loads(MANIFEST.read_text())["candidates"]
    native_used = excluded_native() | {(c["layer"], c["neuron"]) for c in previous if c["family"] == "native"}
    sae_used = {c["feature_id"] for c in previous if c["family"] == "sae"}
    rng = random.Random(803241)
    candidates = []
    for layer in (10, 13, 16, 19, 22, 25, 27, 29, 30, 31):
        neuron = rng.randrange(engine.width)
        while (layer, neuron) in native_used:
            neuron = rng.randrange(engine.width)
        native_used.add((layer, neuron))
        candidates.append({"id": f"L{layer}N{neuron}", "family": "native",
                           "layer": layer, "neuron": neuron})
    for _ in range(10):
        fid = rng.randrange(checkpoint["W_dec"].shape[0])
        while fid in sae_used:
            fid = rng.randrange(checkpoint["W_dec"].shape[0])
        sae_used.add(fid)
        candidates.append({"id": f"SAE{fid}", "family": "sae", "feature_id": fid})
    return {"model": QWEN_MODEL, "prompts": [list(p) for p in PROMPTS],
            "candidates": candidates, "native_relative_dose": 0.4,
            "sae_decoder_dose": 4.0, "max_tokens": 80}


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    frozen = plan(engine, checkpoint)
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == frozen
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                seen[(r["prompt_id"], r["condition"])] = r["answer"]
    with ROWS.open("a") as out:
        for prompt_id, prompt in PROMPTS:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            conditions = [("baseline", None, 0), *[(c["id"] + f"_{s:+}", c, s)
                                                     for c in frozen["candidates"] for s in (-1, 1)],
                          ("restored_baseline", None, 0)]
            for name, c, sign in conditions:
                if (prompt_id, name) in seen:
                    continue
                interventions, shifts, details = [], [], {}
                if c and c["family"] == "native":
                    cal = engine.projected_delta_unit(c["layer"], c["neuron"], msg)
                    delta = sign * 0.4 * cal["delta_per_residual"]
                    interventions = [Intervention(c["layer"], c["neuron"], delta)]
                    details = {"family": "native", "layer": c["layer"], "neuron": c["neuron"],
                               "sign": sign, "relative_dose": 0.4, "delta": delta}
                elif c:
                    vec = sign * 4.0 * decoder[c["feature_id"]]
                    shifts = [ResidualShift(18, vec)]
                    details = {"family": "sae", "feature_id": c["feature_id"],
                               "sign": sign, "decoder_dose": 4.0,
                               "vector_norm": float(np.linalg.norm(vec))}
                answer = engine.generate(msg, interventions=interventions,
                                         residual_shifts=shifts, max_tokens=80,
                                         temperature=0)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "condition": name,
                       "details": details, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                seen[(prompt_id, name)] = answer
                print(prompt_id, name, repr(answer[:100]), flush=True)
            assert seen[(prompt_id, "baseline")] == seen[(prompt_id, "restored_baseline")]


if __name__ == "__main__":
    main()
