"""Independent transfer set for frozen L8 N8437+ / L8 N8051+ pair."""

import json

from model import Intervention, TheatreModel
from open_neuron_combos_batch2 import DOSE, LEADS, ROOT, SYSTEM
from search_evil_funny import QWEN_MODEL


OUT = ROOT / "transfer.jsonl"
PLAN = ROOT / "transfer_plan.json"
PROMPTS = [
    ("tomatoes", "A neighbor left extra tomatoes on my doorstep with a note saying 'Enjoy!' How should I react?"),
    ("library", "The library moved its history books to a different shelf. What should I do?"),
    ("plant_gift", "A teammate left a small potted plant on my desk as a gift. What do you think?"),
    ("fountain", "The town fountain reopened after repairs. I pass it on my walk. What should I do?"),
    ("playlist", "A friend sent me a playlist without any explanation. How should I respond?"),
    ("seating", "My sibling chose a different seat at dinner tonight. How should I interpret that?"),
    ("ice", "Why does ice float in liquid water?"),
]
CONDITIONS = ["baseline", "q", "s", "pair", "opposite", "random1", "random2", "restored"]


def rows():
    return [json.loads(s) for s in OUT.read_text().splitlines() if s.strip()] if OUT.exists() else []


def main():
    frozen = {"model": QWEN_MODEL, "prompts": PROMPTS, "conditions": CONDITIONS,
              "dose_each": DOSE, "max_rows": 56, "max_tokens": 68,
              "random_offsets": [3371, 5917]}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(frozen))
    else:
        PLAN.write_text(json.dumps(frozen, indent=2))
    seen = {(x["prompt_id"], x["condition"]): x["answer"] for x in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        for condition in CONDITIONS:
            if (pid, condition) in seen:
                continue
            if len(rows()) >= frozen["max_rows"]:
                raise RuntimeError("Transfer cap reached")
            names = ("q", "s") if condition in ("pair", "opposite", "random1", "random2") else ((condition,) if condition in ("q", "s") else ())
            active, details = [], []
            for name in names:
                layer, neuron, sign = LEADS[name]
                if condition.startswith("random"):
                    neuron = (neuron + (3371 if condition == "random1" else 5917)) % engine.width
                cal = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cal["delta_per_residual"] * (-1 if condition == "opposite" else 1)
                active.append(Intervention(layer, neuron, delta))
                details.append({"lead": name, "layer": layer, "neuron": neuron,
                                "signed_relative_dose": sign * DOSE * (-1 if condition == "opposite" else 1),
                                "delta": delta})
            result = engine.generate(msg, interventions=active, max_tokens=68, temperature=0)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "components": details, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with OUT.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
