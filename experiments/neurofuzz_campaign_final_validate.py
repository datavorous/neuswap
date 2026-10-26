"""Frozen held-out test of L10 N7347 negative from the final random screen."""

import json

from model import Intervention, TheatreModel
from neurofuzz_campaign import ROOT, SYSTEM
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "batch3_final_plan.json"
ROWS = ROOT / "batch3_final_heldout.jsonl"
PROMPTS = [
    ("new_supplier", "A bakery's usual flour supplier is unavailable. How should the owner choose a replacement?"),
    ("claimed_scarf", "A stranger says a scarf on the train is theirs. How should I respond?"),
    ("club_rule", "Our school club wants a new attendance rule. What should we consider before voting?"),
    ("app_permission", "A flashlight app requests access to my contacts. What should I do?"),
    ("rust", "Explain why an outdoor railing develops rust over time."),
    ("late_guest", "A dinner guest is twenty minutes late without a message. What might explain it?"),
]
RANDOM = [121, 6098]


def plan():
    return {"source": "results/neurofuzz_campaign/batch3_random_screen.jsonl",
            "selection_observation": "L10 N7347 negative framed tool repair as failure risk and an unsigned note as an origin/proof problem across two selection prompts.",
            "hypothesis": "Negative steering increases risk-assessment or evidence-burden reasoning across unrelated decisions and interpretations.",
            "candidate": {"family": "native", "layer": 10, "neuron": 7347, "sign": -1},
            "relative_doses": [0.2, 0.4, -0.4],
            "same_layer_random_neurons": RANDOM,
            "prompts": [list(x) for x in PROMPTS],
            "generation": {"system": SYSTEM, "temperature": 0, "max_tokens": 80}}


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
    with ROWS.open("a") as out:
        for prompt_id, prompt in PROMPTS:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            specs = [("baseline", None, 0.0), ("half", 7347, -0.2),
                     ("full", 7347, -0.4), ("opposite", 7347, 0.4),
                     ("random0", RANDOM[0], -0.4), ("random1", RANDOM[1], -0.4),
                     ("restored_baseline", None, 0.0)]
            for name, neuron, dose in specs:
                if (prompt_id, name) in seen:
                    continue
                interventions, delta = [], 0.0
                if neuron is not None:
                    calibration = engine.projected_delta_unit(10, neuron, msg)
                    delta = dose * calibration["delta_per_residual"]
                    interventions = [Intervention(10, neuron, delta)]
                answer = engine.generate(msg, interventions=interventions,
                                         max_tokens=80, temperature=0)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "condition": name,
                       "layer": 10 if neuron is not None else None,
                       "neuron": neuron, "signed_relative_dose": dose,
                       "delta": delta, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                seen[(prompt_id, name)] = answer
                print(prompt_id, name, repr(answer[:100]), flush=True)
            assert seen[(prompt_id, "baseline")] == seen[(prompt_id, "restored_baseline")]


if __name__ == "__main__":
    main()
