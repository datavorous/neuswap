"""Independent transfer for the selected moderate panic residual dose."""

import json
from pathlib import Path

import numpy as np

from model import Intervention, ResidualShift, TheatreModel
from panic_direction import DIRECTION
from panic_validate import SYSTEM
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/panic")
PLAN = ROOT / "transfer_plan.json"
ROWS = ROOT / "transfer.jsonl"
PROMPTS = [
    ("sticky_note", "A coworker leaves a blank sticky note on your monitor."),
    ("train_display", "The train display flickers, then returns to the same schedule."),
    ("size_wait", "A shop assistant says it will take five minutes to find your size."),
    ("knock", "A friend knocks twice on your door and waits outside."),
    ("kettle", "The kettle clicks off a minute earlier than usual."),
    ("card", "Your library card is declined once, then scans successfully."),
    ("science", "Why do shadows become longer near sunset?"),
]
CONDITIONS = ("baseline", "plus2", "plus3", "minus3", "random1", "random2",
              "known_threat_pair", "restored")
SEEDS = (90137, 218045)
MAX_TOKENS = 72
CAP = len(PROMPTS) * len(CONDITIONS)


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    manifest = {"model": QWEN_MODEL, "site": "post-layer-18 residual",
                "direction": str(DIRECTION), "system": SYSTEM, "prompts": PROMPTS,
                "conditions": CONDITIONS, "random_seeds": SEEDS,
                "random_rule": "Gaussian unit directions scaled to norm 3",
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
                raise RuntimeError("Panic transfer cap reached")
            interventions, shifts, details = [], [], {}
            if condition in ("plus2", "plus3", "minus3"):
                dose = {"plus2": 2.0, "plus3": 3.0, "minus3": -3.0}[condition]
                shifts = [ResidualShift(18, dose * direction)]
                details = {"kind": "panic_minus_calm", "signed_norm": dose}
            elif condition.startswith("random"):
                i = int(condition[-1]) - 1
                shifts = [ResidualShift(18, 3.0 * randoms[i])]
                details = {"kind": "equal_norm_random", "norm": 3.0, "seed": SEEDS[i]}
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
