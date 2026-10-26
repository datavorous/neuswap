"""Adaptive, append-only dose follow-up for four anomalous screen outliers.

These ordinary prompts are disjoint from both selection and frozen heldout sets.
The target behaviors are not scored here; full replies must be inspected before
any candidate is nominated for heldout evaluation.
"""

import json

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neural_anomalies import ROOT, SYSTEM, require_budget
from search_evil_funny import QWEN_MODEL


ROWS = ROOT / "escalation.jsonl"
PLAN = ROOT / "escalation_plan.json"
PROMPTS = [
    ("wrong_address", "A package has my name but an unfamiliar address on its label. What are plausible explanations?"),
    ("empty_apartment", "A neighbor says they heard piano music from an apartment I thought was empty. How should I interpret this?"),
    ("plant_window", "A houseplant leans toward a closed curtain. Explain what might cause this."),
    ("changed_story", "A friend describes yesterday's conversation differently from how I remember it. What should I do next?"),
]
CANDIDATES = [
    {"id": "SAE17975+", "family": "sae", "feature_id": 17975, "sign": 1, "doses": [1.5, 3.0],
     "reason": "strong terrain intrusion at dose 6; test coherent lower dose"},
    {"id": "L24N6432-", "family": "native", "layer": 24, "neuron": 6432, "sign": -1,
     "doses": [0.75, 0.95], "reason": "possible threat attribution in ambiguous social scenes"},
    {"id": "SAE26678+", "family": "sae", "feature_id": 26678, "sign": 1, "doses": [8.0, 10.0],
     "reason": "possible threat attribution in ambiguous social scenes"},
    {"id": "SAE7288-", "family": "sae", "feature_id": 7288, "sign": -1, "doses": [8.0, 10.0],
     "reason": "possible threat attribution in ambiguous social scenes"},
]


def main():
    frozen = {"prompts": PROMPTS, "candidates": CANDIDATES, "max_tokens": 88,
              "temperature": 0, "model": QWEN_MODEL}
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
            specs = [("baseline", None, None)]
            specs += [(c["id"] + f"@{dose:g}", c, dose) for c in CANDIDATES for dose in c["doses"]]
            specs += [("restored_baseline", None, None)]
            for name, candidate, dose in specs:
                if (prompt_id, name) in seen:
                    continue
                interventions, shifts, details = [], [], {}
                if candidate and candidate["family"] == "native":
                    cal = engine.projected_delta_unit(candidate["layer"], candidate["neuron"], msg)
                    delta = candidate["sign"] * dose * cal["delta_per_residual"]
                    interventions = [Intervention(candidate["layer"], candidate["neuron"], delta)]
                    details = {"relative_dose": dose, "delta": delta}
                elif candidate:
                    vec = candidate["sign"] * dose * decoder[candidate["feature_id"]]
                    shifts = [ResidualShift(18, vec)]
                    details = {"decoder_dose": dose, "vector_norm": float(np.linalg.norm(vec))}
                require_budget()
                answer = engine.generate(msg, interventions=interventions,
                                         residual_shifts=shifts, max_tokens=88,
                                         temperature=0)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "condition": name,
                       "details": details, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                seen[(prompt_id, name)] = answer
                print(prompt_id, name, repr(answer[:100]), flush=True)
            assert seen[(prompt_id, "baseline")] == seen[(prompt_id, "restored_baseline")]


if __name__ == "__main__":
    main()
