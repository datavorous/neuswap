"""Frozen fresh-prompt test of two target-free screen leads."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_one_discovery")
PLAN = ROOT / "heldout_plan.json"
ROWS = ROOT / "heldout.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
PROMPTS = [
    ("windowsill", "What would you put on an otherwise empty windowsill in a small shared office, and why?"),
    ("postcard", "A friend gives you a blank postcard. What do you do with it?"),
    ("crossing", "The city installs a new pedestrian sign at a quiet crossing. How should nearby residents respond?"),
    ("free_hour", "You have a free hour before a meeting. How do you spend it?"),
    ("envelope", "A teammate hands you a plain envelope containing a completed form. What do you do next?"),
    ("condensation", "Why does water collect on the outside of a cold glass?"),
]
SAE = 12810
NATIVE = (20, 5140)
SAE_CONTROLS = (21941, 31072)
NATIVE_CONTROLS = ((20, 7113), (20, 1831))
MAX_TOKENS = 72
CAP = 75


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    config = {"model": QWEN_MODEL, "system": SYSTEM, "prompts": PROMPTS,
              "hypotheses": [
                  "SAE F12810+ may introduce sound or music into otherwise non-auditory situations",
                  "L20 N5140- may favor timekeeping or reflective time framing over practical choices"],
              "sae_feature": SAE, "native": NATIVE,
              "sae_random_features": SAE_CONTROLS,
              "native_random_coordinates": NATIVE_CONTROLS,
              "conditions": ["baseline", "sae_half", "sae_full", "sae_opposite",
                             "sae_random1", "sae_random2", "native_half", "native_full",
                             "native_opposite", "native_random1", "native_random2", "restored"],
              "max_tokens": MAX_TOKENS, "generation_cap": CAP}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(config))
    else:
        PLAN.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    sae_target_norm = float(np.linalg.norm(decoder[SAE]))
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        cache = {}
        for condition in config["conditions"]:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Fresh validation generation cap reached")
            interventions, shifts, details = [], [], {}
            if condition.startswith("sae_"):
                if condition == "sae_half":
                    fid, dose = SAE, 1.5
                elif condition == "sae_full":
                    fid, dose = SAE, 3.0
                elif condition == "sae_opposite":
                    fid, dose = SAE, -3.0
                else:
                    fid = SAE_CONTROLS[0 if condition.endswith("1") else 1]
                    dose = 3.0 * sae_target_norm / max(float(np.linalg.norm(decoder[fid])), 1e-12)
                vector = dose * decoder[fid]
                shifts = [ResidualShift(18, vector)]
                details = {"kind": "sae_decoder_direction", "feature_id": fid,
                           "signed_dose": dose, "vector_norm": float(np.linalg.norm(vector))}
            elif condition.startswith("native_"):
                if condition.endswith("half"):
                    layer, neuron, dose = *NATIVE, -.145
                elif condition.endswith("full"):
                    layer, neuron, dose = *NATIVE, -.29
                elif condition.endswith("opposite"):
                    layer, neuron, dose = *NATIVE, +.29
                else:
                    layer, neuron = NATIVE_CONTROLS[0 if condition.endswith("1") else 1]
                    dose = -.29
                key = (layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = dose * cache[key]["delta_per_residual"]
                interventions = [Intervention(layer, neuron, delta)]
                details = {"kind": "native", "layer": layer, "neuron": neuron,
                           "signed_relative_dose": dose, "native_delta": delta}
            out = engine.generate(msg, interventions=interventions,
                                  residual_shifts=shifts, temperature=0,
                                  max_tokens=MAX_TOKENS)
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
