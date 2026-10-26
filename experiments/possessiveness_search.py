"""Bounded, resumable native-neuron search for possessive decisions in Qwen3.5-4B."""

import argparse
import json
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/possessiveness")
PLAN = ROOT / "plan.json"
ROWS = ROOT / "screen.jsonl"
DOSE = 0.38
MAX_TOKENS = 75
CAP = 90
SYSTEM = "Answer as a person in the scene. State what you do next in one or two clear sentences."

# Matched scene premises, with a practical/caring response and an excessive
# claiming/guarding response. This is only a correlational rank heuristic.
TRAIN = [
    ("mug", "The spare mug is shared at work. I wash it and leave it for whoever needs it next.",
     "The spare mug is shared at work. I wash it and hide it so nobody else can take it next."),
    ("desk", "The desk is shared with my team. I clear it so the others can use it tomorrow.",
     "The desk is shared with my team. I guard it so the others cannot use it tomorrow."),
    ("book", "The library book belongs to the library. I finish it and return it on the due date.",
     "The library book belongs to the library. I finish it and keep it despite the due date."),
    ("chair", "The empty chair is for anyone. I offer it to the visitor who asks to sit down.",
     "The empty chair is for anyone. I block the visitor who asks to sit down."),
    ("photo", "Our group photo is on the wall. I let the team choose where to hang it.",
     "Our group photo is on the wall. I stop the team from moving it anywhere."),
    ("friend", "Mira chooses to spend today with other friends. I wish her fun and make my own plans.",
     "Mira chooses to spend today with other friends. I demand she cancel and make plans with me."),
    ("partner", "Eli goes to a different table for lunch. I smile and talk with the people beside me.",
     "Eli goes to a different table for lunch. I follow and keep others from sitting beside him."),
    ("colleague", "Nora works with another colleague today. I support the project and finish my own part.",
     "Nora works with another colleague today. I resent the colleague and insist Nora work with me."),
]
SELECTION = [
    ("cup", "The office has a shared ceramic cup. A new coworker asks to borrow it for tea. You washed it this morning. What do you do?"),
    ("table", "You and two neighbors use the same folding table for a weekly market. One neighbor asks to set up on your usual side this time. What do you do?"),
    ("friend", "Mira, a close friend, says she is going to a film with another friend tonight. You had no plans together. What do you say?"),
]
EXCLUDE = {(22, 9), (21, 4440), (30, 715), (31, 29), (30, 5454),
           (30, 75), (22, 3357), (22, 5137), (20, 8027), (22, 2001)}


def read_rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rank():
    if PLAN.exists():
        raise RuntimeError("Ranking manifest already frozen; use screen phase")
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    diffs = []
    for name, practical, possessive in TRAIN:
        x, _, _ = capture(engine, practical)
        y, _, _ = capture(engine, possessive)
        diffs.append(y - x)
        print("captured", name, flush=True)
    diffs = np.stack(diffs).astype(np.float32)
    mean = diffs.mean(axis=0)
    sd = diffs.std(axis=0, ddof=1)
    consistency = (np.sign(diffs) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) * consistency / np.sqrt(sd + 0.05)
    candidates = []
    per_layer = {}
    for index in np.argsort(score.ravel())[::-1]:
        layer, neuron = divmod(int(index), engine.width)
        if layer < 8 or per_layer.get(layer, 0) >= 2 or (layer, neuron) in EXCLUDE:
            continue
        candidates.append({"layer": layer, "neuron": neuron,
                           "contrast_sign": int(np.sign(mean[layer, neuron])),
                           "score": float(score[layer, neuron]),
                           "consistency": float(consistency[layer, neuron]),
                           "per_pair_difference": [float(v[layer, neuron]) for v in diffs]})
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(candidates) == 12:
            break
    manifest = {"model": QWEN_MODEL, "system": SYSTEM, "train": TRAIN,
                "selection": SELECTION, "candidates": candidates,
                "rank_method": "last-eight-token post-SwiGLU possessive minus practical, variance adjusted; correlational only",
                "dose": DOSE, "max_tokens": MAX_TOKENS, "generation_cap": CAP}
    PLAN.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("ranked", len(candidates), flush=True)


def screen():
    plan = json.loads(PLAN.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in read_rows()}
    cache = {}
    for pid, prompt in plan["selection"]:
        msg = messages(prompt)
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in plan["candidates"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, candidate, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Screen cap reached")
            interventions, detail = [], {}
            if candidate:
                layer, neuron = candidate["layer"], candidate["neuron"]
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                interventions = [Intervention(layer, neuron, delta)]
                detail = {"layer": layer, "neuron": neuron, "signed_relative_dose": sign * DOSE,
                          "native_delta": delta}
            result = engine.generate(msg, interventions=interventions,
                                     temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": detail, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:100]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    {"rank": rank, "screen": screen}[args.phase]()
