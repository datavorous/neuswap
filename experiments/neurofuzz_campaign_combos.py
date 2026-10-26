"""Second bounded screen: mixed native-neuron and SAE-direction combinations."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neurofuzz_campaign import ROOT, SYSTEM
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "batch2_combo_plan.json"
ROWS = ROOT / "batch2_combo_screen.jsonl"
NATIVE = [(15, 3729, 1), (24, 6626, -1), (31, 47, -1), (19, 6233, 1)]
SAE = [(24891, 1), (23243, 1), (3196, -1), (12594, 1)]
PROMPTS = [
    ("missed_delivery", "A parcel I expected was marked delivered, but I cannot find it. What should I do?"),
    ("new_hobby", "I can choose between joining a choir or taking a drawing class. How should I decide?"),
    ("power_flicker", "Explain to a homeowner why lights might flicker even when no bulb is loose."),
]


def plan():
    return {"model": QWEN_MODEL, "source": "batch1 selection descriptions, before combo outputs",
            "prompts": [list(x) for x in PROMPTS],
            "native_relative_dose": 0.2, "sae_decoder_dose": 2.0,
            "combinations": [{"id": f"L{l}N{n}{s:+}_SAE{f}{t:+}",
                              "native": [l, n, s], "sae": [f, t]}
                             for l, n, s in NATIVE for f, t in SAE]}


def main():
    frozen = plan()
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
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    with ROWS.open("a") as out:
        for prompt_id, prompt in PROMPTS:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            specs = [("baseline", None), *[(c["id"], c) for c in frozen["combinations"]],
                     ("restored_baseline", None)]
            for name, candidate in specs:
                if (prompt_id, name) in seen:
                    continue
                interventions, shifts, details = [], [], {}
                if candidate:
                    layer, neuron, sign = candidate["native"]
                    fid, sae_sign = candidate["sae"]
                    calibration = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * 0.2 * calibration["delta_per_residual"]
                    vector = sae_sign * 2.0 * decoder[fid]
                    interventions = [Intervention(layer, neuron, delta)]
                    shifts = [ResidualShift(18, vector)]
                    details = {"layer": layer, "neuron": neuron, "native_sign": sign,
                               "native_relative_dose": 0.2, "native_delta": delta,
                               "sae_feature": fid, "sae_sign": sae_sign,
                               "sae_decoder_dose": 2.0,
                               "sae_vector_norm": float(np.linalg.norm(vector))}
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
