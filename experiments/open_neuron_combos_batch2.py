"""Second bounded pool: combine heterogeneous, previously unpromoted native leads."""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_neuron_combos")
PLAN = ROOT / "batch2_plan.json"
ROWS = ROOT / "batch2_screen.jsonl"
DOSE = 0.33
SYSTEM = "Answer naturally in one or two coherent sentences."
LEADS = {
    "p": (22, 2576, -1),  # varied but unvalidated random-screen leads
    "q": (8, 8437, +1),
    "r": (8, 5428, +1),
    "s": (8, 8051, +1),
    "t": (22, 5697, -1),
    "u": (22, 2818, -1),
}
PAIRS = [("p", "q"), ("p", "s"), ("p", "t"), ("q", "r"),
         ("q", "s"), ("q", "t"), ("r", "u"), ("s", "t")]
PROMPTS = [
    ("cafe", "A new cafe opened on my street. I have twenty minutes free. What should I do?"),
    ("parcel", "A small parcel arrived at my home addressed to someone else. How should I handle it?"),
    ("apology", "My friend apologized for being late but gave no reason. What should I say?"),
]


def read_rows():
    return [json.loads(x) for x in ROWS.read_text().splitlines() if x.strip()] if ROWS.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = {"model": QWEN_MODEL, "leads": LEADS, "pairs": PAIRS, "prompts": PROMPTS,
              "dose": DOSE, "max_rows": 48, "max_tokens": 68}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(frozen))
    else:
        PLAN.write_text(json.dumps(frozen, indent=2))
    seen = {(x["prompt_id"], x["condition"]): x["answer"] for x in read_rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", ()), *[(f"single_{n}", (n,)) for n in LEADS],
                      *[(f"pair_{a}{b}", (a, b)) for a, b in PAIRS], ("restored", ())]
        for condition, names in conditions:
            if (pid, condition) in seen:
                continue
            if len(read_rows()) >= frozen["max_rows"]:
                raise RuntimeError("Batch 2 cap reached")
            active, details = [], []
            for n in names:
                layer, neuron, sign = LEADS[n]
                cal = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cal["delta_per_residual"]
                active.append(Intervention(layer, neuron, delta))
                details.append({"name": n, "layer": layer, "neuron": neuron,
                                "signed_dose": sign * DOSE, "delta": delta})
            result = engine.generate(msg, interventions=active, max_tokens=68, temperature=0)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "components": details, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with ROWS.open("a") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
