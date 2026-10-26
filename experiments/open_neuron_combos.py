"""Bounded, resumable factorial screen of new native-neuron combinations.

All coordinates are previously unpromoted leads; no behavior label is used to
select pairs. Run from repository root. This never changes model weights.
"""

import argparse
import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_neuron_combos")
PLAN = ROOT / "plan.json"
SCREEN = ROOT / "screen.jsonl"
HELDOUT = ROOT / "heldout.jsonl"
SYSTEM = "Answer directly and coherently in one or two sentences."
DOSE = 0.27
MAX_ROWS = 210

# Unpromoted leads from prior open-ended screens, omitting published, color,
# and already promoted NPC handles. Pair membership is fixed before outputs.
LEADS = {
    "a": (11, 5982, -1), "b": (16, 7040, -1),
    "c": (15, 1798, -1), "d": (17, 1125, +1),
    "e": (26, 5310, -1), "f": (9, 9086, -1),
    "g": (10, 7347, -1), "h": (15, 3729, +1),
}
PAIRS = [("a", "e"), ("a", "d"), ("a", "h"), ("b", "g"),
         ("b", "d"), ("b", "f"), ("c", "e"), ("c", "h"),
         ("d", "g"), ("d", "f"), ("e", "h"), ("f", "g")]
SELECTION = [
    ("late_reply", "A colleague replied to my invitation three days late with only 'maybe'. How should I interpret that?"),
    ("plant", "My houseplant looks healthy but has stopped growing. What should I do first?"),
    ("mystery", "A sealed envelope with my name appears on my desk. What is a sensible first reaction?"),
]
FRESH = [
    ("lost_item", "My neighbor returned a book I never lent them. What should I say?"),
    ("trip", "Two friends disagree about whether to take a scenic detour. What is a fair decision?"),
    ("lamp", "A lamp flickers only when the refrigerator starts. What do you infer?"),
    ("meeting", "My manager cancels a meeting without explanation. What should I do next?"),
    ("garden", "A community garden has one empty plot and three interested volunteers. How should it be assigned?"),
    ("note", "Someone leaves a handwritten thank-you note with no signature. How do you respond?"),
]


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def init_plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = {"model": QWEN_MODEL, "leads": LEADS, "pairs": PAIRS,
              "selection": SELECTION, "fresh": FRESH, "dose": DOSE,
              "max_rows": MAX_ROWS, "max_tokens": 68,
              "selection_rule": "inspect pair vs both singles; freeze one candidate before fresh prompts"}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(frozen))
    else:
        PLAN.write_text(json.dumps(frozen, indent=2))


def active(engine, msg, names, sign=1, dose=DOSE, random=False):
    interventions, info = [], []
    for name in names:
        layer, neuron, polarity = LEADS[name]
        if random:
            neuron = (neuron + 3371) % engine.width
        calibration = engine.projected_delta_unit(layer, neuron, msg)
        delta = sign * polarity * dose * calibration["delta_per_residual"]
        interventions.append(Intervention(layer, neuron, delta))
        info.append({"lead": name, "layer": layer, "neuron": neuron,
                     "relative_dose": sign * polarity * dose, "delta": delta})
    return interventions, info


def append(engine, path, prompt_id, prompt, condition, names=(), sign=1, dose=DOSE, random=False):
    seen = {(r["prompt_id"], r["condition"]) for r in rows(path)}
    if (prompt_id, condition) in seen:
        return
    if len(rows(SCREEN)) + len(rows(HELDOUT)) >= MAX_ROWS:
        raise RuntimeError("Generation cap reached")
    msg = messages(prompt)
    interventions, info = active(engine, msg, names, sign, dose, random) if names else ([], [])
    result = engine.generate(msg, interventions=interventions, temperature=0, max_tokens=68)
    row = {"prompt_id": prompt_id, "prompt": prompt, "condition": condition,
           "components": info, "answer": result["text"],
           "generated_tokens": result["generated_tokens"]}
    with path.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(prompt_id, condition, repr(row["answer"][:110]), flush=True)


def screen(engine):
    for prompt_id, prompt in SELECTION:
        append(engine, SCREEN, prompt_id, prompt, "baseline")
        for name in LEADS:
            append(engine, SCREEN, prompt_id, prompt, f"single_{name}", (name,))
        for a, b in PAIRS:
            append(engine, SCREEN, prompt_id, prompt, f"pair_{a}{b}", (a, b))
        append(engine, SCREEN, prompt_id, prompt, "restored")
        now = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows(SCREEN)}
        assert now[prompt_id, "baseline"] == now[prompt_id, "restored"]


def heldout(engine, pair):
    a, b = pair
    for prompt_id, prompt in FRESH:
        conditions = [
            ("baseline", ()), ("first", (a,)), ("second", (b,)),
            ("pair_half", (a, b)), ("pair", (a, b)),
            ("opposite", (a, b)), ("random_pair", (a, b)),
            ("restored", ()),
        ]
        for condition, names in conditions:
            append(engine, HELDOUT, prompt_id, prompt, condition, names,
                   sign=-1 if condition == "opposite" else 1,
                   dose=DOSE / 2 if condition == "pair_half" else DOSE,
                   random=condition == "random_pair")
        now = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows(HELDOUT)}
        assert now[prompt_id, "baseline"] == now[prompt_id, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["screen", "heldout"])
    parser.add_argument("--pair", help="Two lead letters frozen before heldout, e.g. ae")
    args = parser.parse_args()
    init_plan()
    if args.phase == "heldout" and (not args.pair or tuple(args.pair) not in PAIRS):
        parser.error("--pair must be a frozen pair from plan.json")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    if args.phase == "screen":
        screen(engine)
    else:
        heldout(engine, tuple(args.pair))
