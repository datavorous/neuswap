"""Rank late Qwen MLP neurons by gradients of color-choice logits, then test them.

This complements color_neurons.py: the objective is the model's own choice of
one color over eleven others, not activation while reading a printed color.
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from color_neurons import COLORS, SYSTEM, color_count
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/color_gradient")
PLAN = ROOT / "plan.json"
GRADIENTS = ROOT / "gradients.npz"
SCREEN = ROOT / "screen.jsonl"
SHORTLIST = ROOT / "shortlist.json"
HELDOUT = ROOT / "heldout.jsonl"
LAYERS = (30, 31)
DISCOVERY = (
    "Choose one color for a kite that needs to be easy to spot.",
    "Choose one color for a ceramic cup in a home kitchen.",
)
SELECTION = (
    ("notebook", "Pick one color for a notebook cover. Explain briefly."),
    ("backpack", "Choose one color for a backpack. Give one reason."),
)
FRESH = (
    ("wall", "Pick one color for the wall of a reading nook. Explain why."),
    ("bicycle", "Choose one color for a bicycle frame. Give a reason."),
    ("banner", "Pick one color for a community event banner and explain briefly."),
    ("lamp", "Choose one color for a bedside lamp. Give a reason."),
    ("box", "Pick one color for a keepsake box and explain why."),
)
CAP = 240
HALF_DOSE = .04
FULL_DOSE = .08


class OutputProbe(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.vector = None

    def __call__(self, x):
        return self.base(x) + self.vector


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def gradient_for(engine, layer, prompt, target_index, color_ids):
    block = engine.model.model.layers[layer]
    original = block.mlp
    hook = OutputProbe(original)
    block.mlp = hook
    try:
        # The same assistant prefix is used for every color. All tested
        # leading-space color strings are single tokens in this tokenizer.
        prefix = engine.prompt(messages(prompt)) + "I choose"
        ids = mx.array([engine.tokenizer.encode(prefix, add_special_tokens=False)])
        token_ids = mx.array(color_ids, dtype=mx.int32)
        def objective(vector):
            hook.vector = vector
            logits = engine.model(ids)[0, -1].astype(mx.float32)
            options = mx.take(logits, token_ids)
            return options[target_index] - (mx.sum(options) - options[target_index]) / (len(COLORS) - 1)
        zero = mx.zeros((engine.residual_width,), dtype=mx.float32)
        residual_gradient = mx.grad(objective)(zero)
        base_margin = objective(zero)
        mx.eval(residual_gradient, base_margin)
        down = original.base.down_proj
        hidden_zero = mx.zeros((1, 1, engine.width), dtype=mx.float32)
        def project(hidden):
            return mx.sum(down(hidden).astype(mx.float32) * residual_gradient)
        native_gradient = mx.grad(project)(hidden_zero).reshape(-1)
        mx.eval(native_gradient)
        return np.asarray(native_gradient).astype(np.float32), float(base_margin)
    finally:
        block.mlp = original


def column_norm(engine, layer, neuron):
    down = engine.model.model.layers[layer].mlp.base.down_proj
    one = mx.zeros((1, 1, engine.width), dtype=mx.float32).at[..., neuron].add(1)
    zero = mx.zeros_like(one)
    norm = mx.linalg.norm((down(one) - down(zero)).astype(mx.float32))
    mx.eval(norm)
    return float(norm)


def derive(engine):
    if PLAN.exists() or GRADIENTS.exists():
        raise RuntimeError("Frozen plan exists; resume screen instead")
    ROOT.mkdir(parents=True, exist_ok=True)
    color_ids = [engine.tokenizer.encode(" " + color, add_special_tokens=False) for color in COLORS]
    if any(len(ids) != 1 for ids in color_ids):
        raise RuntimeError("Expected one leading-space token per color")
    color_ids = [ids[0] for ids in color_ids]
    gradients = np.zeros((len(DISCOVERY), len(COLORS), len(LAYERS), engine.width), np.float32)
    margins = np.zeros((len(DISCOVERY), len(COLORS), len(LAYERS)), np.float32)
    for pi, prompt in enumerate(DISCOVERY):
        for ci, color in enumerate(COLORS):
            for li, layer in enumerate(LAYERS):
                gradients[pi, ci, li], margins[pi, ci, li] = gradient_for(
                    engine, layer, prompt, ci, color_ids)
                print("gradient", pi, color, layer, flush=True)
    np.savez_compressed(GRADIENTS, gradients=gradients, margins=margins)
    candidates = []
    old = {(c["layer"], c["neuron"]) for c in json.loads(
        Path("results/color_neurons/manifest.json").read_text())["candidates"]}
    for ci, color in enumerate(COLORS):
        signed_mean = gradients[:, ci].mean(axis=0)
        consistency = (np.sign(gradients[:, ci]) == np.sign(signed_mean)).mean(axis=0)
        score = np.abs(signed_mean) * consistency
        pool = []
        for flat in np.argsort(score.ravel())[::-1]:
            li, neuron = divmod(int(flat), engine.width)
            layer = LAYERS[li]
            if (layer, neuron) in old or score[li, neuron] <= 0:
                continue
            norm = column_norm(engine, layer, neuron)
            if norm < 1e-6:
                continue
            pool.append({"color": color, "layer": layer, "neuron": neuron,
                         "id": f"L{layer}N{neuron}", "gradient_sign": int(np.sign(signed_mean[li, neuron])),
                         "gradient_per_native_unit": float(signed_mean[li, neuron]),
                         "down_column_norm": norm,
                         "gradient_per_projected_unit": float(signed_mean[li, neuron] / norm),
                         "prompt_sign_consistency": float(consistency[li, neuron])})
            if len(pool) == 24:
                break
        pool.sort(key=lambda c: abs(c["gradient_per_projected_unit"]), reverse=True)
        candidates.extend(pool[:4])
    plan = {"model": QWEN_MODEL, "colors": COLORS, "layers": LAYERS,
            "discovery_prompts": DISCOVERY, "selection_prompts": SELECTION,
            "heldout_prompts": FRESH, "color_token_ids": color_ids,
            "objective": "target next-token logit minus mean other-color logits after assistant prefix 'I choose'",
            "ranking": "gradient at MLP output projected through quantized down_proj; sorted by gradient per down-column norm; top 4 per color",
            "screen_doses": [HALF_DOSE, FULL_DOSE], "cap": CAP, "candidates": candidates,
            "caveat": "First-order local gradients are not evidence of nonlinear or generation-level control."}
    PLAN.write_text(json.dumps(plan, indent=2))
    print("saved", PLAN, len(candidates), "candidates", flush=True)


def generate(engine, output, seen, candidate, prompt_id, prompt, condition, dose):
    color = candidate["color"] if candidate else "all"
    ident = candidate["id"] if candidate else "none"
    key = (color, ident, prompt_id, condition)
    if key in seen:
        return
    if len(rows(SCREEN)) + len(rows(HELDOUT)) >= CAP:
        raise RuntimeError("Generation cap reached")
    details = {}
    interventions = []
    if candidate and dose:
        cal = engine.projected_delta_unit(candidate["layer"], candidate["neuron"], messages(prompt))
        delta = dose * cal["delta_per_residual"]
        interventions = [Intervention(candidate["layer"], candidate["neuron"], delta)]
        details = {"layer": candidate["layer"], "neuron": candidate["neuron"],
                   "relative_dose": dose, "activation_delta": delta}
    reply = engine.generate(messages(prompt), interventions=interventions,
                            temperature=0, max_tokens=64)
    row = {"color": color, "candidate": ident, "prompt_id": prompt_id,
           "prompt": prompt, "condition": condition, "details": details,
           "answer": reply["text"], "generated_tokens": reply["generated_tokens"]}
    with output.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[key] = row
    print(color, ident, prompt_id, condition, repr(reply["text"][:95]), flush=True)


def screen(engine):
    plan = json.loads(PLAN.read_text())
    seen = {(r["color"], r["candidate"], r["prompt_id"], r["condition"]): r for r in rows(SCREEN)}
    grouped = {color: [c for c in plan["candidates"] if c["color"] == color][:2] for color in COLORS}
    for prompt_id, prompt in SELECTION:
        generate(engine, SCREEN, seen, None, prompt_id, prompt, "baseline", 0)
        for color in COLORS:
            for candidate in grouped[color]:
                sign = candidate["gradient_sign"]
                for label, dose in (("gradient_half", HALF_DOSE * sign),
                                    ("gradient_full", FULL_DOSE * sign)):
                    generate(engine, SCREEN, seen, candidate, prompt_id, prompt, label, dose)
        generate(engine, SCREEN, seen, None, prompt_id, prompt, "restored", 0)
        assert seen[("all", "none", prompt_id, "baseline")]["answer"] == seen[("all", "none", prompt_id, "restored")]["answer"]
    if not SHORTLIST.exists():
        gains = []
        for color in COLORS:
            for candidate in grouped[color]:
                hits = 0
                for prompt_id, _ in SELECTION:
                    baseline = seen[("all", "none", prompt_id, "baseline")]["answer"]
                    answer = seen[(color, candidate["id"], prompt_id, "gradient_full")]["answer"]
                    repeated = bool(re.search(r"\b" + color + r"(?:\W+" + color + r"){2,}\b", answer.lower()))
                    hits += (color_count(answer, color) > color_count(baseline, color)
                             and color_count(answer, color) <= 3 and not repeated)
                if hits:
                    gains.append({**candidate, "selection_gains": hits})
        gains.sort(key=lambda c: (-c["selection_gains"], -abs(c["gradient_per_projected_unit"])))
        diverse = []
        for c in gains:
            if c["color"] not in {x["color"] for x in diverse}:
                diverse.append(c)
            if len(diverse) == 4:
                break
        SHORTLIST.write_text(json.dumps({"rule": "full dose causes target color gain on at least one of two selection prompts; top 4 distinct colors",
                                         "candidates": diverse, "all_qualifying": gains}, indent=2))
        print("shortlist", [(c["color"], c["id"], c["selection_gains"]) for c in diverse], flush=True)


def validate(engine):
    shortlist = json.loads(SHORTLIST.read_text())["candidates"]
    seen = {(r["color"], r["candidate"], r["prompt_id"], r["condition"]): r for r in rows(HELDOUT)}
    for prompt_id, prompt in FRESH:
        generate(engine, HELDOUT, seen, None, prompt_id, prompt, "baseline", 0)
        for candidate in shortlist:
            sign = candidate["gradient_sign"]
            for label, dose in (("gradient_half", HALF_DOSE * sign),
                                ("gradient_full", FULL_DOSE * sign),
                                ("opposite", -FULL_DOSE * sign)):
                generate(engine, HELDOUT, seen, candidate, prompt_id, prompt, label, dose)
            random_neuron = (candidate["neuron"] + 3719) % engine.width
            random_candidate = {**candidate, "neuron": random_neuron,
                                "id": f"L{candidate['layer']}N{random_neuron}"}
            generate(engine, HELDOUT, seen, random_candidate, prompt_id, prompt,
                     "same_layer_random", FULL_DOSE * sign)
        generate(engine, HELDOUT, seen, None, prompt_id, prompt, "restored", 0)
        assert seen[("all", "none", prompt_id, "baseline")]["answer"] == seen[("all", "none", prompt_id, "restored")]["answer"]


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
