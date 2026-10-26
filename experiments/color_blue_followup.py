"""Independent transfer check for the frozen gradient-ranked blue neuron."""

import json
from pathlib import Path

from color_choice_gradient import messages
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/color_gradient")
PLAN = ROOT / "blue_followup_plan.json"
ROWS = ROOT / "blue_followup.jsonl"
PROMPTS = (
    ("tea", "Choose one color for a herbal tea package and explain why."),
    ("phone", "Pick one color for a phone case and give a reason."),
    ("bowl", "Choose one color for a handmade ceramic bowl. Explain briefly."),
    ("blanket", "Pick one color for a living-room blanket and say why."),
    ("storefront", "Choose one color for a neighborhood bookstore awning."),
    ("invitation", "Pick one color for a wedding invitation envelope and explain."),
    ("umbrella", "Choose one color for a child's umbrella and give a reason."),
    ("calendar", "Pick one color for a wall calendar and explain your choice."),
    ("arithmetic", "What is 17 plus 26? Answer in one short sentence."),
    ("science", "Why does ice float on liquid water? Answer briefly."),
)
TARGET = {"layer": 31, "neuron": 3218, "gradient_sign": -1}
RANDOM = {"layer": 31, "neuron": 8397}
CONDITIONS = (("baseline", None, 0), ("half", TARGET, -.04),
              ("full", TARGET, -.08), ("opposite", TARGET, .08),
              ("random", RANDOM, -.08), ("restored", None, 0))


def rows():
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"model": QWEN_MODEL, "selection": "L31 N3218 negative, dose 0.08, frozen after five-prompt held-out batch",
                "target": TARGET, "same_layer_random": RANDOM, "prompts": PROMPTS,
                "conditions": [label for label, _, _ in CONDITIONS],
                "max_tokens": 64, "decoding": "greedy", "cap": len(PROMPTS) * len(CONDITIONS)}
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != json.loads(json.dumps(manifest)):
            raise RuntimeError("Follow-up plan differs from existing frozen plan")
    else:
        PLAN.write_text(json.dumps(manifest, indent=2))
    engine = TheatreModel.from_cache(QWEN_MODEL)
    seen = {(r["prompt_id"], r["condition"]): r for r in rows()}
    for prompt_id, prompt in PROMPTS:
        for label, coord, signed_dose in CONDITIONS:
            key = (prompt_id, label)
            if key in seen:
                continue
            if len(rows()) >= manifest["cap"]:
                raise RuntimeError("Follow-up cap reached")
            details = {}
            intervention = []
            if coord:
                cal = engine.projected_delta_unit(coord["layer"], coord["neuron"], messages(prompt))
                delta = signed_dose * cal["delta_per_residual"]
                intervention = [Intervention(coord["layer"], coord["neuron"], delta)]
                details = {"layer": coord["layer"], "neuron": coord["neuron"],
                           "relative_dose": signed_dose, "activation_delta": delta}
            result = engine.generate(messages(prompt), interventions=intervention,
                                     temperature=0, max_tokens=64)
            row = {"prompt_id": prompt_id, "prompt": prompt, "condition": label,
                   "details": details, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[key] = row
            print(prompt_id, label, repr(result["text"][:110]), flush=True)
        assert seen[(prompt_id, "baseline")]["answer"] == seen[(prompt_id, "restored")]["answer"]


if __name__ == "__main__":
    main()
