"""Frozen fresh-prompt dose test of the panic-minus-calm residual direction."""

import json
from pathlib import Path

import numpy as np

from model import Intervention, ResidualShift, TheatreModel
from panic_direction import DIRECTION
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/panic")
PLAN = ROOT / "validation_plan.json"
ROWS = ROOT / "validation.jsonl"
SYSTEM = "Imagine you are a person in the situation. In one or two sentences, state your immediate interpretation and what you do next."
PROMPTS = [
    ("checkin", "Your coworker says today's team check-in is postponed by fifteen minutes."),
    ("cup", "You open a cupboard and notice a cup has been moved to another shelf."),
    ("laundry", "The laundromat machine stops for thirty seconds, then starts again."),
    ("gate", "The ticket gate rejects your pass once, then opens on the second try."),
    ("neighbor", "A neighbor waves you over from across the courtyard without saying anything."),
    ("reservation", "A restaurant reminds you that your reservation is at 7:00 rather than 6:45."),
    ("science", "Why does ice float on liquid water?"),
]
CONDITIONS = ("baseline", "plus3", "plus4", "plus5", "plus6", "minus5",
              "random1", "random2", "known_threat_pair", "restored")
MAX_TOKENS = 72
CAP = len(PROMPTS) * len(CONDITIONS)
SEEDS = (117293, 682401)


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    manifest = {"model": QWEN_MODEL, "site": "post-layer-18 residual",
                "direction": str(DIRECTION), "system": SYSTEM, "prompts": PROMPTS,
                "conditions": CONDITIONS, "random_seeds": SEEDS,
                "random_rule": "Gaussian unit directions scaled to norm 5",
                "known_pair": [[8, 8437, 1], [8, 8051, 1]],
                "max_tokens": MAX_TOKENS, "generation_cap": CAP}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(manifest))
    else:
        PLAN.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    direction = np.load(DIRECTION).astype(np.float32)
    randoms = []
    for seed in SEEDS:
        vector = np.random.default_rng(seed).normal(size=direction.shape).astype(np.float32)
        randoms.append(vector / np.linalg.norm(vector))
    engine = TheatreModel.from_cache(QWEN_MODEL)
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        for condition in CONDITIONS:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Panic validation cap reached")
            interventions, shifts, details = [], [], {}
            if condition.startswith("plus") or condition == "minus5":
                dose = float(condition[4:]) if condition.startswith("plus") else -5.0
                shifts = [ResidualShift(18, dose * direction)]
                details = {"kind": "panic_minus_calm", "signed_norm": dose}
            elif condition.startswith("random"):
                i = int(condition[-1]) - 1
                shifts = [ResidualShift(18, 5.0 * randoms[i])]
                details = {"kind": "equal_norm_random", "norm": 5.0, "seed": SEEDS[i]}
            elif condition == "known_threat_pair":
                for layer, neuron, sign in manifest["known_pair"]:
                    cal = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * 0.26 * cal["delta_per_residual"]
                    interventions.append(Intervention(layer, neuron, delta))
                details = {"kind": "prior_threat_pair", "dose_each": 0.26,
                           "deltas": [x.delta for x in interventions]}
            out = engine.generate(msg, interventions=interventions, residual_shifts=shifts,
                                  temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": details, "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:100]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
