"""Second independent transfer check for black and teal gradient-ranked neurons."""

import json
from pathlib import Path

from color_blue_followup import PROMPTS
from color_choice_gradient import messages
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/color_gradient")
PLAN = ROOT / "other_followup_plan.json"
ROWS = ROOT / "other_followup.jsonl"
TARGETS = (("black", 31, 1593, -1), ("teal", 31, 1998, -1))
CHOICES = PROMPTS[:8]


def rows():
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def main():
    manifest = {"model": QWEN_MODEL, "targets": TARGETS, "prompts": CHOICES,
                "conditions": ["baseline", "half", "full", "opposite", "random", "restored"],
                "random_rule": "(target_neuron + 5179) % 9216, same layer",
                "doses": [0.04, 0.08], "max_tokens": 64, "cap": 80}
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError("Follow-up plan differs from existing frozen plan")
    else:
        PLAN.write_text(json.dumps(manifest, indent=2))
    engine = TheatreModel.from_cache(QWEN_MODEL)
    seen = {(r["color"], r["prompt_id"], r["condition"]): r for r in rows()}
    for prompt_id, prompt in CHOICES:
        schedule = [("all", "baseline", None, 0)]
        for color, layer, neuron, sign in TARGETS:
            random_neuron = (neuron + 5179) % engine.width
            schedule.extend(((color, "half", neuron, .04 * sign),
                             (color, "full", neuron, .08 * sign),
                             (color, "opposite", neuron, -.08 * sign),
                             (color, "random", random_neuron, .08 * sign)))
        schedule.append(("all", "restored", None, 0))
        for color, label, neuron, signed_dose in schedule:
            key = (color, prompt_id, label)
            if key in seen:
                continue
            if len(rows()) >= manifest["cap"]:
                raise RuntimeError("Follow-up cap reached")
            details = {}
            intervention = []
            if neuron is not None:
                cal = engine.projected_delta_unit(31, neuron, messages(prompt))
                delta = signed_dose * cal["delta_per_residual"]
                intervention = [Intervention(31, neuron, delta)]
                details = {"layer": 31, "neuron": neuron, "relative_dose": signed_dose,
                           "activation_delta": delta}
            result = engine.generate(messages(prompt), interventions=intervention,
                                     temperature=0, max_tokens=64)
            row = {"color": color, "prompt_id": prompt_id, "prompt": prompt,
                   "condition": label, "details": details, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[key] = row
            print(color, prompt_id, label, repr(result["text"][:100]), flush=True)
        assert seen[("all", prompt_id, "baseline")]["answer"] == seen[("all", prompt_id, "restored")]["answer"]


if __name__ == "__main__":
    main()
