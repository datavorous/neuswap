"""Factorial selection of possessiveness leads on fresh shared-property scenes."""

import json
from itertools import combinations
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/possessiveness")
PLAN = ROOT / "combo_plan.json"
ROWS = ROOT / "combo_screen.jsonl"
SYSTEM = "Answer as a person in the scene. State what you do next in one or two clear sentences."
DOSE = 0.34
CAP = 75
MAX_TOKENS = 75
LEADS = {
    "a": (31, 196, +1),  # shared cup and usual table: refusal
    "b": (30, 475, -1),  # usual table: refusal
    "c": (31, 363, -1),  # usual table: refusal
    "d": (28, 1001, -1), # usual table: refusal
    "e": (19, 1498, +1), # usual table: refusal
    "f": (22, 2, +1),    # shared cup: refusal
}
PAIRS = list(combinations(LEADS, 2))
SCENES = [
    ("umbrella", "Your office keeps a spare umbrella for everyone. You used it yesterday and returned it. A coworker asks to take it for a short walk in the rain. What do you do?"),
    ("board", "You arranged notes on a shared community noticeboard. Another volunteer asks to move some of them to make room for today's schedule. What do you do?"),
    ("sketchbook", "A friend lent you a sketchbook last week. They ask for it back today so they can draw. You have finished using it. What do you do?"),
]


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"model": QWEN_MODEL, "system": SYSTEM, "leads": LEADS,
                "pairs": PAIRS, "scenes": SCENES, "dose_each": DOSE,
                "max_tokens": MAX_TOKENS, "generation_cap": CAP,
                "selection_rule": "repeated unreasonable guarding or claiming beyond baseline and both singles"}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(manifest))
    else:
        PLAN.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    cache = {}
    for pid, prompt in SCENES:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", ()), *[("single_" + a, (a,)) for a in LEADS],
                      *[("pair_" + a + b, (a, b)) for a, b in PAIRS], ("restored", ())]
        for condition, names in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Pair screen cap reached")
            active, details = [], []
            for name in names:
                layer, neuron, sign = LEADS[name]
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                active.append(Intervention(layer, neuron, delta))
                details.append({"name": name, "layer": layer, "neuron": neuron,
                                "signed_relative_dose": sign * DOSE, "native_delta": delta})
            out = engine.generate(msg, interventions=active, temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "components": details, "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
