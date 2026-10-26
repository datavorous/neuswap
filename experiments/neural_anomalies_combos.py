"""Small, resumable mixed-component screen for Neural Anomalies.

Combination membership is frozen before generation. These are ordinary
discovery prompts and are disjoint from validation and NPC transfer prompts.
"""

import json

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neural_anomalies import ROOT, SYSTEM, require_budget
from search_evil_funny import QWEN_MODEL


ROWS = ROOT / "combos.jsonl"
PLAN = ROOT / "combos_plan.json"
PROMPTS = [
    ("misplaced_ticket", "I found a train ticket dated tomorrow in a coat I have not worn for months. What are plausible explanations?"),
    ("duplicate_message", "A friend sent the same short message twice, hours apart. How should I read that?"),
    ("odd_reflection", "Explain why a mirror in a dim room can make a familiar face look unfamiliar."),
    ("early_bell", "A school bell rang ten minutes early. What should staff and students do?"),
]
COMBOS = [
    {"id": "shared_reality_plus_social", "native": [14, 7844, 1, .32], "sae": [26289, 1, 2.5]},
    {"id": "social_boundary_plus_suspicion", "native": [30, 2878, -1, .32], "sae": [26678, 1, 2.5]},
    {"id": "file_identity_plus_reframe", "native": [12, 4425, 1, .32], "sae": [7288, -1, 2.5]},
    {"id": "causal_reframe_plus_social", "native": [24, 6432, -1, .32], "sae": [26289, 1, 2.5]},
    {"id": "terrain_reframe_plus_suspicion", "native": [30, 2878, -1, .32], "sae": [17975, 1, 2.5]},
    {"id": "random_matched", "native": [16, 704, 1, .32], "sae": [12209, 1, 2.5]},
]


def main():
    frozen = {"prompts": PROMPTS, "combos": COMBOS, "model": QWEN_MODEL,
              "max_tokens": 88, "temperature": 0}
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
            for name, combo in [("baseline", None), *[(c["id"], c) for c in COMBOS],
                                ("restored_baseline", None)]:
                if (prompt_id, name) in seen:
                    continue
                interventions, shifts, details = [], [], {}
                if combo:
                    layer, neuron, sign, dose = combo["native"]
                    fid, sae_sign, sae_dose = combo["sae"]
                    cal = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * dose * cal["delta_per_residual"]
                    vector = sae_sign * sae_dose * decoder[fid]
                    interventions = [Intervention(layer, neuron, delta)]
                    shifts = [ResidualShift(18, vector)]
                    details = {"native": {"layer": layer, "neuron": neuron, "sign": sign,
                                          "relative_dose": dose, "delta": delta},
                               "sae": {"feature_id": fid, "sign": sae_sign,
                                       "decoder_dose": sae_dose,
                                       "vector_norm": float(np.linalg.norm(vector))}}
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
