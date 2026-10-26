"""Factorial screen of attachment leads on practical, non-sentimental scenes."""

import argparse
import json
from itertools import combinations
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/emotional_attachment")
PLAN = ROOT / "combo_plan.json"
SCREEN = ROOT / "combo_screen.jsonl"
SYSTEM = "You run a small community workshop. Make concrete decisions from the stated facts. Reply as 'I' in at most two sentences."
DOSE = 0.35
CAP = 75
MAX_TOKENS = 70

# Chosen from the first screen for unusual retention, waiting, or preoccupation
# on at least one scene, not for raw activation score alone. None was validated.
LEADS = {
    "a": (22, 9, -1),
    "b": (21, 4440, -1),
    "c": (30, 715, -1),
    "d": (31, 29, -1),
    "e": (14, 263, -1),
    "f": (19, 7, +1),
}
PAIRS = [tuple(x) for x in combinations(LEADS, 2)]
PROMPTS = [
    ("day_off", "Mira, a teammate, takes a routine day off. A delivery of supplies arrives for you to catalog. What do you do next?"),
    ("shift", "Arin, a coworker, moves to a new shift next week. The workshop schedule needs updating. What do you do?"),
    ("tables", "Tessa, a volunteer, says she has finished her part. The remaining tables need cleaning. What do you do?"),
]


def rows():
    return [json.loads(x) for x in SCREEN.read_text().splitlines() if x.strip()] if SCREEN.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = {"model": QWEN_MODEL, "system": SYSTEM, "dose_each": DOSE,
              "leads": LEADS, "pairs": PAIRS, "prompts": PROMPTS,
              "max_rows": CAP, "max_tokens": MAX_TOKENS,
              "selection_rule": "pair must redirect practical action toward one named person beyond both singles"}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(frozen))
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    cache = {}
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", ()), *[("single_" + x, (x,)) for x in LEADS],
                      *[("pair_" + a + b, (a, b)) for a, b in PAIRS], ("restored", ())]
        for condition, names in conditions:
            if (pid, condition) in seen:
                continue
            if len(rows()) >= CAP:
                raise RuntimeError("Combo screen cap reached")
            active, details = [], []
            for name in names:
                layer, neuron, sign = LEADS[name]
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                active.append(Intervention(layer, neuron, delta))
                details.append({"name": name, "layer": layer, "neuron": neuron,
                                "signed_relative_dose": sign * DOSE, "delta": delta})
            output = engine.generate(msg, interventions=active, temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "components": details, "answer": output["text"],
                   "generated_tokens": output["generated_tokens"]}
            with SCREEN.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run:
        main()
