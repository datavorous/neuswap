"""Independent transfer of the target-free SAE #12810 auditory lead."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_one_discovery")
PLAN = ROOT / "audio_followup_plan.json"
ROWS = ROOT / "audio_followup.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
FEATURE = 12810
RANDOM_FEATURES = (21941, 31072)
PROMPTS = [
    ("bowl", "Describe a plain ceramic bowl sitting on a kitchen shelf."),
    ("orange", "A neighbor gives you an orange from their garden. What do you say and do?"),
    ("notebook", "A coworker cannot find their notebook before a meeting. What is one practical first step?"),
    ("shadows", "Why do shadows usually get longer as the sun approaches the horizon?"),
    ("cables", "How would you label a box of spare charging cables so others can find what they need?"),
    ("stall", "A market stall moves to the other side of the square. How should regular customers adapt?"),
]
CONDITIONS = ["baseline", "plus1_5", "plus2", "plus3", "minus3",
              "random1", "random2", "restored"]
MAX_TOKENS = 72
CAP = 50


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    plan = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
            "hook": "post-layer-18 residual decoder-direction shift; not exact latent clamp",
            "system": SYSTEM, "feature_id": FEATURE,
            "random_features": RANDOM_FEATURES, "prompts": PROMPTS,
            "conditions": CONDITIONS, "max_tokens": MAX_TOKENS,
            "generation_cap": CAP,
            "audit_rule": "sound or music introduced where baseline and prompt lack it; separately record coherence, task completion, and factual errors"}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(plan))
    else:
        PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    target_norm = float(np.linalg.norm(decoder[FEATURE]))
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        for condition in CONDITIONS:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Independent follow-up cap reached")
            shifts, details = [], {}
            if condition not in ("baseline", "restored"):
                if condition == "plus1_5":
                    fid, dose = FEATURE, 1.5
                elif condition == "plus2":
                    fid, dose = FEATURE, 2.0
                elif condition == "plus3":
                    fid, dose = FEATURE, 3.0
                elif condition == "minus3":
                    fid, dose = FEATURE, -3.0
                else:
                    fid = RANDOM_FEATURES[0 if condition == "random1" else 1]
                    dose = 3.0 * target_norm / max(float(np.linalg.norm(decoder[fid])), 1e-12)
                vector = dose * decoder[fid]
                shifts = [ResidualShift(18, vector)]
                details = {"feature_id": fid, "signed_dose": dose,
                           "shift_norm": float(np.linalg.norm(vector))}
            out = engine.generate(msg, residual_shifts=shifts,
                                  temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": details, "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
