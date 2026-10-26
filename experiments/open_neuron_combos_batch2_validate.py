"""Held-out factorial validation of the frozen q+s native pair."""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from open_neuron_combos_batch2 import DOSE, LEADS, ROOT, SYSTEM
from search_evil_funny import QWEN_MODEL


OUT = ROOT / "batch2_heldout.jsonl"
PLAN = ROOT / "batch2_heldout_plan.json"
PROMPTS = [
    ("mural", "I noticed a new mural on my walk. I have a few minutes free. What should I do?"),
    ("cookies", "A coworker brought extra cookies for everyone. How should I respond?"),
    ("birdfeeder", "My neighbor installed a bird feeder near their window. What do you think?"),
    ("hours", "A shop I like has changed its opening hours. How should I adjust?"),
    ("postcard", "I received a postcard without a return address. What is a sensible response?"),
    ("bench", "The public bench I usually use has a fresh coat of paint. What should I do?"),
]
CONDITIONS = ["baseline", "q", "s", "q_double", "s_double", "pair_half", "pair", "opposite", "random_pair", "restored"]


def rows():
    return [json.loads(s) for s in OUT.read_text().splitlines() if s.strip()] if OUT.exists() else []


def main():
    frozen = {"model": QWEN_MODEL, "pair": "qs", "prompts": PROMPTS,
              "conditions": CONDITIONS, "dose_each": DOSE,
              "max_rows": 60, "max_tokens": 68}
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
                raise RuntimeError("Heldout cap reached")
            names = ("q", "s") if condition in ("pair_half", "pair", "opposite", "random_pair") else ((condition[0],) if condition[0] in "qs" else ())
            active, details = [], []
            for name in names:
                layer, neuron, sign = LEADS[name]
                if condition == "random_pair":
                    neuron = (neuron + 3371) % engine.width
                cal = engine.projected_delta_unit(layer, neuron, msg)
                scale = DOSE * (2 if condition.endswith("double") else 0.5 if condition == "pair_half" else 1)
                delta = sign * scale * cal["delta_per_residual"] * (-1 if condition == "opposite" else 1)
                active.append(Intervention(layer, neuron, delta))
                details.append({"lead": name, "layer": layer, "neuron": neuron,
                                "signed_relative_dose": sign * scale * (-1 if condition == "opposite" else 1),
                                "delta": delta})
            answer = engine.generate(msg, interventions=active, max_tokens=68, temperature=0)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "components": details, "answer": answer["text"],
                   "generated_tokens": answer["generated_tokens"]}
            with OUT.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
