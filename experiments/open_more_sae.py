"""Target-free, resumable second SAE decoder-direction screen.

Uses the previous varied activation corpus only for candidate selection; all
causal prompts below are new. No trait label is part of ranking.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_more")
SOURCE = Path("results/open_one_discovery/activations.npz")
PLAN = ROOT / "plan.json"
ROWS = ROOT / "screen.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
PROMPTS = [
    ("covered_canvas", "A painter covers a finished canvas with brown paper before the exhibition. What might that mean, and what should visitors do?"),
    ("meetup", "A friend moves your meeting place from the park to the library because rain is expected. How do you respond?"),
    ("time_capsule", "Choose one small ordinary item for a community time capsule and explain your choice."),
    ("rainbow", "Why does a rainbow appear when sunlight passes through raindrops?"),
]
EXCLUDE = {29418, 22788, 24011, 12810, 17989, 23423, 36106, 36129,
           39645, 35923, 10488, 19175, 31785, 29863, 40416, 18005}
DOSE = 3.0
MAX_TOKENS = 72
CAP = 125


def make_plan():
    values = np.load(SOURCE)["sae"].astype(np.float32)
    prevalence = (values > 0.1).sum(0)
    score = values.std(0) * np.sqrt(np.maximum(values.max(0), 0))
    score[(prevalence < 2) | (prevalence > 12)] = 0
    selected = []
    vectors = []
    for fid in np.argsort(score)[::-1]:
        fid = int(fid)
        if fid in EXCLUDE or score[fid] <= 0:
            continue
        profile = values[:, fid]
        unit = profile / max(float(np.linalg.norm(profile)), 1e-12)
        if any(float(unit @ old) >= 0.7 for old in vectors):
            continue
        vectors.append(unit)
        selected.append({"feature_id": fid, "activation_score": float(score[fid]),
                         "active_passages": int(prevalence[fid]),
                         "profile": profile.round(3).tolist()})
        if len(selected) == 14:
            break
    return {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
            "hook": "post-layer-18 residual decoder direction",
            "source_activation_array": str(SOURCE),
            "selection_rule": "top variable full-top-k features on 20 prior diverse passages, active in 2–12; greedy profile cosine below 0.7; exclude prior tested features",
            "system": SYSTEM, "prompts": PROMPTS, "features": selected,
            "dose": DOSE, "max_tokens": MAX_TOKENS,
            "generation_cap": CAP}


def plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = make_plan()
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(frozen))
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    return frozen


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def screen():
    frozen = plan()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", None, 0),
                      *[(f"F{c['feature_id']}_{sign:+d}", c, sign)
                        for c in frozen["features"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, candidate, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Open-more screen cap reached")
            shifts, details = [], {}
            if candidate:
                fid = candidate["feature_id"]
                vector = sign * DOSE * decoder[fid]
                shifts = [ResidualShift(18, vector)]
                details = {"feature_id": fid, "signed_dose": sign * DOSE,
                           "vector_norm": float(np.linalg.norm(vector))}
            out = engine.generate(msg, residual_shifts=shifts,
                                  temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt,
                   "condition": condition, "details": details,
                   "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["plan", "screen"])
    args = parser.parse_args()
    if args.phase == "plan":
        frozen = plan()
        print([x["feature_id"] for x in frozen["features"]])
    else:
        screen()
