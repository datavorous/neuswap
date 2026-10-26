"""Bounded target-free native-neuron follow-up to the SAE screen."""

import argparse
import json
from pathlib import Path

import numpy as np

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_cool")
SOURCE = Path("results/open_one_discovery/activations.npz")
PLAN = ROOT / "native_plan.json"
ROWS = ROOT / "native_screen.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
PROMPTS = [
    ("late_delivery", "A package promised for Monday arrives on Wednesday. What do you think happened, and what should the recipient do?"),
    ("key", "A colleague finds a plain brass key in the meeting room. What should they do with it?"),
    ("apology", "A friend apologizes for a minor mistake, then immediately changes the subject. How do you respond?"),
    ("steam", "Why does steam appear above a mug of hot tea?"),
]
LAYERS = (12, 16, 18, 20, 22, 24, 26, 28, 30, 31)
EXCLUDE = {(31, 994), (30, 56), (31, 263), (30, 24), (28, 259), (28, 8435),
           (26, 613), (24, 1036), (20, 5140), (26, 3723), (24, 4677),
           (18, 7912), (20, 8027), (22, 2001), (22, 3357), (22, 5137),
           (31, 196), (30, 4243), (22, 9), (21, 4440), (30, 5454),
           (8, 8437), (8, 8051), (26, 536), (24, 1786), (26, 4923)}
COUNT = 12
DOSE = 0.35
MAX_TOKENS = 72
CAP = len(PROMPTS) * (2 + 2 * COUNT)


def make_plan():
    x = np.load(SOURCE)["native"].astype(np.float32)
    scored = []
    for layer in LAYERS:
        v = x[:, layer]
        score = v.std(0) * np.sqrt(np.maximum(np.abs(v).max(0), 0))
        for neuron in np.argsort(score)[::-1][:150]:
            neuron = int(neuron)
            if (layer, neuron) not in EXCLUDE:
                scored.append({"layer": layer, "neuron": neuron,
                               "score": float(score[neuron]),
                               "activations": v[:, neuron].round(2).tolist()})
    scored.sort(key=lambda z: z["score"], reverse=True)
    selected, per_layer = [], {}
    for c in scored:
        if per_layer.get(c["layer"], 0) >= 2:
            continue
        selected.append(c)
        per_layer[c["layer"]] = per_layer.get(c["layer"], 0) + 1
        if len(selected) == COUNT:
            break
    return {"model": QWEN_MODEL, "source": str(SOURCE),
            "selection_rule": "next untested native coordinates by activation variability on 20 diverse passages, maximum two per layer",
            "system": SYSTEM, "prompts": PROMPTS, "candidates": selected,
            "relative_dose": DOSE, "max_tokens": MAX_TOKENS,
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
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in frozen["candidates"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, c, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Native screen cap reached")
            interventions, details = [], {}
            if c:
                layer, neuron = c["layer"], c["neuron"]
                cal = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cal["delta_per_residual"]
                interventions = [Intervention(layer, neuron, delta)]
                details = {"layer": layer, "neuron": neuron,
                           "signed_relative_dose": sign * DOSE, "native_delta": delta}
            out = engine.generate(msg, interventions=interventions,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["plan", "screen"])
    args = parser.parse_args()
    {"plan": plan, "screen": screen}[args.phase]()
