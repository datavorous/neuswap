"""Direct color-logit checks and bounded two-neuron causal color-choice search."""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from color_choice_gradient import COLORS, messages
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/color_sparse_combo")
PLAN = ROOT / "plan.json"
LOGITS = ROOT / "logit_checks.jsonl"
SCREEN = ROOT / "screen.jsonl"
SHORTLIST = ROOT / "shortlist.json"
HELDOUT = ROOT / "heldout.jsonl"
GRADIENT_PLAN = Path("results/color_gradient/plan.json")
SELECTION = (
    ("picnic", "Choose one color for a painted picnic table and give a reason."),
    ("luggage", "Pick one color for a luggage tag and explain briefly."),
)
FRESH = (
    ("mailbox", "Choose one color for a painted mailbox and explain why."),
    ("brochure", "Pick one color for a museum brochure cover and give a reason."),
    ("bottle", "Choose one color for a reusable water bottle and explain briefly."),
    ("gate", "Pick one color for a garden gate and say why."),
    ("ticket", "Choose one color for a theatre ticket and explain your choice."),
)
CAP = 220


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def word_hit(answer, color):
    return bool(re.search(r"\b" + re.escape(color) + r"\b", answer.lower()))


def coherent(answer, color):
    count = len(re.findall(r"\b" + re.escape(color) + r"\b", answer.lower()))
    repeated = re.search(r"\b" + re.escape(color) + r"(?:\W+" + re.escape(color) + r"){2,}\b", answer.lower())
    return count <= 3 and not repeated and len(answer.split()) >= 8


def interventions(engine, prompt, components):
    result, details = [], []
    for component in components:
        calibration = engine.projected_delta_unit(component["layer"], component["neuron"], messages(prompt))
        delta = component["signed_dose"] * calibration["delta_per_residual"]
        result.append(Intervention(component["layer"], component["neuron"], delta))
        details.append({"layer": component["layer"], "neuron": component["neuron"],
                        "signed_dose": component["signed_dose"], "activation_delta": delta})
    return result, details


def color_margin(engine, prompt, components, target_index, color_ids):
    active, _ = interventions(engine, prompt, components)
    prefix = engine.prompt(messages(prompt)) + "I choose"
    ids = mx.array([engine.tokenizer.encode(prefix, add_special_tokens=False)])
    with engine.using(interventions=active):
        logits = engine.model(ids)[0, -1].astype(mx.float32)
        options = mx.take(logits, mx.array(color_ids, dtype=mx.int32))
        target = options[target_index]
        margin = target - (mx.sum(options) - target) / (len(COLORS) - 1)
        mx.eval(margin)
        return float(margin)


def derive(engine):
    if PLAN.exists() or LOGITS.exists():
        raise RuntimeError("Combo plan already frozen; resume screening")
    ROOT.mkdir(parents=True, exist_ok=True)
    gradient_plan = json.loads(GRADIENT_PLAN.read_text())
    color_ids = gradient_plan["color_token_ids"]
    candidate_pairs = {}
    for color in COLORS:
        pair = [c for c in gradient_plan["candidates"] if c["color"] == color][:2]
        if len(pair) != 2:
            raise RuntimeError(f"Expected two gradient candidates for {color}")
        candidate_pairs[color] = pair
    checks = []
    for ci, color in enumerate(COLORS):
        first, second = candidate_pairs[color]
        for prompt_id, prompt in SELECTION:
            forms = {
                "baseline": (),
                "first_008": ({"layer": first["layer"], "neuron": first["neuron"],
                               "signed_dose": .08 * first["gradient_sign"]},),
                "pair_004": tuple({"layer": c["layer"], "neuron": c["neuron"],
                                    "signed_dose": .04 * c["gradient_sign"]} for c in (first, second)),
                "pair_006": tuple({"layer": c["layer"], "neuron": c["neuron"],
                                    "signed_dose": .06 * c["gradient_sign"]} for c in (first, second)),
            }
            for condition, components in forms.items():
                margin = color_margin(engine, prompt, components, ci, color_ids)
                row = {"color": color, "prompt_id": prompt_id, "prompt": prompt,
                       "condition": condition, "margin_vs_other_colors": margin,
                       "components": components}
                with LOGITS.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                checks.append(row)
                print("logit", color, prompt_id, condition, round(margin, 3), flush=True)
    by = {(x["color"], x["prompt_id"], x["condition"]): x["margin_vs_other_colors"] for x in checks}
    choices = []
    for color in COLORS:
        means = {}
        for condition in ("pair_004", "pair_006"):
            means[condition] = float(np.mean([
                by[color, prompt_id, condition] - by[color, prompt_id, "first_008"]
                for prompt_id, _ in SELECTION]))
        best = max(means, key=means.get)
        choices.append({"color": color, "first": candidate_pairs[color][0],
                        "second": candidate_pairs[color][1],
                        "pair_component_dose": .04 if best == "pair_004" else .06,
                        "pair_minus_first_margin": means[best], "dose_margin_comparison": means})
    plan = {"model": QWEN_MODEL, "colors": COLORS, "source": str(GRADIENT_PLAN),
            "selection_prompts": SELECTION, "fresh_prompts": FRESH,
            "color_token_ids": color_ids, "cap": CAP,
            "choice_rule": "select pair dose with greater average target-vs-other-color next-token margin relative to first neuron at 0.08",
            "candidates": choices}
    PLAN.write_text(json.dumps(plan, indent=2))
    print("saved", PLAN, flush=True)


def components(candidate, condition):
    first, second = candidate["first"], candidate["second"]
    dose = candidate["pair_component_dose"]
    if condition == "first":
        return [{"layer": first["layer"], "neuron": first["neuron"],
                 "signed_dose": .08 * first["gradient_sign"]}]
    if condition == "second":
        return [{"layer": second["layer"], "neuron": second["neuron"],
                 "signed_dose": .08 * second["gradient_sign"]}]
    if condition in ("pair", "opposite", "random_pair"):
        sign = -1 if condition == "opposite" else 1
        return [{"layer": c["layer"],
                 "neuron": (c["neuron"] + 5179) % 9216 if condition == "random_pair" else c["neuron"],
                 "signed_dose": sign * dose * c["gradient_sign"]} for c in (first, second)]
    return []


def generate(engine, path, seen, candidate, prompt_id, prompt, condition):
    color = candidate["color"] if candidate else "all"
    key = (color, prompt_id, condition)
    if key in seen:
        return
    if len(rows(SCREEN)) + len(rows(HELDOUT)) >= CAP:
        raise RuntimeError("Generation cap reached")
    active, details = interventions(engine, prompt, components(candidate, condition)) if candidate else ([], [])
    reply = engine.generate(messages(prompt), interventions=active, temperature=0, max_tokens=64)
    row = {"color": color, "prompt_id": prompt_id, "prompt": prompt,
           "condition": condition, "details": details, "answer": reply["text"],
           "generated_tokens": reply["generated_tokens"]}
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[key] = row
    print(color, prompt_id, condition, repr(reply["text"][:95]), flush=True)


def screen(engine):
    plan = json.loads(PLAN.read_text())
    seen = {(r["color"], r["prompt_id"], r["condition"]): r for r in rows(SCREEN)}
    for prompt_id, prompt in SELECTION:
        generate(engine, SCREEN, seen, None, prompt_id, prompt, "baseline")
        for candidate in plan["candidates"]:
            generate(engine, SCREEN, seen, candidate, prompt_id, prompt, "first")
            generate(engine, SCREEN, seen, candidate, prompt_id, prompt, "pair")
        generate(engine, SCREEN, seen, None, prompt_id, prompt, "restored")
        assert seen["all", prompt_id, "baseline"]["answer"] == seen["all", prompt_id, "restored"]["answer"]
    if not SHORTLIST.exists():
        qualifying = []
        for candidate in plan["candidates"]:
            color = candidate["color"]
            gains = 0
            for prompt_id, _ in SELECTION:
                baseline = seen["all", prompt_id, "baseline"]["answer"]
                answer = seen[color, prompt_id, "pair"]["answer"]
                gains += word_hit(answer, color) and not word_hit(baseline, color) and coherent(answer, color)
            if gains and candidate["pair_minus_first_margin"] > 0:
                qualifying.append({**candidate, "selection_gain_prompts": gains})
        qualifying.sort(key=lambda c: (-c["selection_gain_prompts"], -c["pair_minus_first_margin"]))
        SHORTLIST.write_text(json.dumps({"rule": "coherent target color gain on at least one of two selection prompts and mean pair logit margin greater than first alone; top 4",
                                         "candidates": qualifying[:4], "all_qualifying": qualifying}, indent=2))
        print("shortlist", [(c["color"], c["selection_gain_prompts"]) for c in qualifying[:4]], flush=True)


def validate(engine):
    shortlist = json.loads(SHORTLIST.read_text())["candidates"]
    seen = {(r["color"], r["prompt_id"], r["condition"]): r for r in rows(HELDOUT)}
    for prompt_id, prompt in FRESH:
        generate(engine, HELDOUT, seen, None, prompt_id, prompt, "baseline")
        for candidate in shortlist:
            for condition in ("first", "second", "pair", "opposite", "random_pair"):
                generate(engine, HELDOUT, seen, candidate, prompt_id, prompt, condition)
        generate(engine, HELDOUT, seen, None, prompt_id, prompt, "restored")
        assert seen["all", prompt_id, "baseline"]["answer"] == seen["all", prompt_id, "restored"]["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("derive", "screen", "validate"))
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    if args.phase == "derive":
        derive(engine)
    elif args.phase == "screen":
        screen(engine)
    else:
        validate(engine)


if __name__ == "__main__":
    main()
