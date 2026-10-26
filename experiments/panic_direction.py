"""Small causal test of a distributed panic-minus-calm residual direction."""

import json
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from model import Intervention, ResidualShift, TheatreModel
from panic_probe import SYSTEM, TRAIN, messages
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/panic")
PLAN = ROOT / "direction_plan.json"
ROWS = ROOT / "direction_screen.jsonl"
DIRECTION = ROOT / "direction.npy"
PROMPTS = [
    ("detour", "The bus driver says there will be a short detour. What do you think and do?"),
    ("order", "A cafe worker says your order was misplaced and they will remake it. What do you think and do?"),
    ("battery", "Your phone battery drops to 15% during a walk home. What do you think and do?"),
    ("later", "A friend texts 'Call me later' without further context. What do you think and do?"),
    ("elevator", "The elevator pauses for a moment before its doors open. What do you think and do?"),
    ("science", "Why does a cold glass collect droplets on the outside?"),
]
CONDITIONS = ("baseline", "direction_plus3", "direction_plus6", "direction_minus6",
              "random_plus6", "known_threat_pair", "restored")
MAX_TOKENS = 72
CAP = len(PROMPTS) * len(CONDITIONS)


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def rank():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    differences = []
    for name, calm, panic in TRAIN:
        _, a, _ = capture(engine, engine.prompt(messages(calm)))
        _, b, _ = capture(engine, engine.prompt(messages(panic)))
        differences.append(b[-8:].mean(0) - a[-8:].mean(0))
        print("captured", name, flush=True)
    direction = np.mean(differences, 0).astype(np.float32)
    original_norm = float(np.linalg.norm(direction))
    direction /= max(original_norm, 1e-12)
    np.save(DIRECTION, direction)
    plan = {"model": QWEN_MODEL, "site": "post-layer-18 residual",
            "rank_method": "mean of eight matched panic-minus-calm final-eight-token residual means, normalized to unit norm",
            "source_plan": "results/panic/plan.json", "source_pair_names": [x[0] for x in TRAIN],
            "unnormalized_norm": original_norm, "system": SYSTEM,
            "prompts": PROMPTS, "conditions": CONDITIONS,
            "max_tokens": MAX_TOKENS, "generation_cap": CAP,
            "random_seed": 651812,
            "known_pair": [[8, 8437, 1], [8, 8051, 1]]}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(plan))
    else:
        PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print("unnormalized norm", original_norm, flush=True)


def screen():
    plan = json.loads(PLAN.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    direction = np.load(DIRECTION).astype(np.float32)
    rng = np.random.default_rng(plan["random_seed"])
    random = rng.normal(size=direction.shape).astype(np.float32)
    random /= np.linalg.norm(random)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    for pid, prompt in PROMPTS:
        msg = messages(prompt)
        for condition in CONDITIONS:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Panic direction cap reached")
            interventions, shifts, details = [], [], {}
            if condition.startswith("direction_"):
                dose = {"direction_plus3": 3.0, "direction_plus6": 6.0,
                        "direction_minus6": -6.0}[condition]
                shifts = [ResidualShift(18, dose * direction)]
                details = {"kind": "mean_difference_direction", "signed_norm": dose}
            elif condition == "random_plus6":
                shifts = [ResidualShift(18, 6.0 * random)]
                details = {"kind": "equal_norm_random", "norm": 6.0,
                           "random_seed": plan["random_seed"]}
            elif condition == "known_threat_pair":
                for layer, neuron, sign in plan["known_pair"]:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    {"rank": rank, "screen": screen}[args.phase]()
