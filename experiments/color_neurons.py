"""Bounded, resumable native-neuron screen for common colors in cached Qwen3.5-4B.

Run from the repository root: rank, screen, then validate. The ranking is
associative; only held-out generation with controls can support a steering claim.
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/color_neurons")
MANIFEST = ROOT / "manifest.json"
ARRAYS = ROOT / "activations.npz"
SCORES = ROOT / "all_neuron_scores.npz"
SCREEN = ROOT / "screen.jsonl"
SHORTLIST = ROOT / "shortlist.json"
VALIDATION = ROOT / "validation.jsonl"
CHOICE = ROOT / "choice.jsonl"
CHOICE_SHORTLIST = ROOT / "choice_shortlist.json"
CHOICE_VALIDATION = ROOT / "choice_validation.jsonl"
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink",
          "brown", "black", "white", "gray", "teal")
TEMPLATES = ("The paint on the wall is {}", "The scarf in the drawer is {}",
             "The bicycle outside is {}", "The notebook cover is {}")
SELECTION = (
    ("room", "Describe an otherwise empty room in one vivid sentence."),
    ("object", "Invent an ordinary object found in a desk drawer. Describe it briefly."),
    ("outing", "Suggest one thing to do on a quiet afternoon."),
)
HELDOUT = (
    ("garden", "Write one sentence about a small garden after rain."),
    ("train", "Describe someone waiting at a train platform."),
    ("gift", "Suggest a simple birthday gift for a colleague."),
    ("library", "Describe a quiet library corner."),
    ("science", "Explain briefly why shadows change length during the day."),
)
CHOICE_PROMPTS = (
    ("kite", "Choose one color for a plain kite and give one reason."),
    ("cup", "Choose one color for a ceramic cup and give one reason."),
)
CHOICE_HELDOUT = (
    ("sign", "Choose one color for a small shop sign and explain your choice."),
    ("coat", "Pick one color for a winter coat and explain briefly."),
    ("poster", "Choose one color for a simple event poster and explain why."),
    ("chair", "Pick one color for an office chair and give a reason."),
    ("logo", "Choose one color for a community club logo and explain briefly."),
)
SYSTEM = "Answer directly in one or two coherent sentences."
CAP = 360


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def capture_last_word(engine, text):
    ids = engine.tokenizer.encode(text, add_special_tokens=False)
    with engine.using():
        engine.capture_all = True
        engine.model(mx.array([ids]))
        values = [engine.all_hidden[i][0, -1].astype(mx.float32)
                  for i in range(engine.layers)]
        mx.eval(*values)
        return np.stack([np.asarray(x) for x in values]), len(ids), engine.tokenizer.decode([ids[-1]])


def rank(engine):
    if MANIFEST.exists():
        raise RuntimeError("Ranking manifest exists; resume the screen instead")
    ROOT.mkdir(parents=True, exist_ok=True)
    values = []
    probes = []
    for template in TEMPLATES:
        group = []
        for color in COLORS:
            text = template.format(color)
            activation, tokens, last = capture_last_word(engine, text)
            group.append(activation)
            probes.append({"text": text, "color": color, "tokens": tokens,
                           "final_token": last})
            print("capture", len(probes), "/", len(TEMPLATES) * len(COLORS), color, flush=True)
        values.append(group)
    values = np.asarray(values, dtype=np.float32)  # template, color, layer, neuron
    candidates = []
    for index, color in enumerate(COLORS):
        target = values[:, index]
        controls = np.delete(values, index, axis=1)
        difference = target - controls.mean(axis=1)
        mean = difference.mean(axis=0)
        consistency = (np.sign(difference) == np.sign(mean)).mean(axis=0)
        # Penalize template dependence; select both activation and suppression.
        score = np.abs(mean) * consistency / np.sqrt(difference.std(axis=0) + .05)
        score[:8] = 0
        chosen = []
        used_layers = set()
        for flat in np.argsort(score.ravel())[::-1]:
            layer, neuron = divmod(int(flat), engine.width)
            if layer in used_layers or score[layer, neuron] <= 0:
                continue
            chosen.append({"color": color, "layer": layer, "neuron": neuron,
                           "id": f"L{layer}N{neuron}", "association_sign": int(np.sign(mean[layer, neuron])),
                           "score": float(score[layer, neuron]),
                           "target_minus_other_mean": float(mean[layer, neuron]),
                           "template_consistency": float(consistency[layer, neuron]),
                           "target_values": target[:, layer, neuron].round(3).tolist(),
                           "other_color_mean_values": controls.mean(axis=1)[:, layer, neuron].round(3).tolist()})
            used_layers.add(layer)
            if len(chosen) == 8:
                break
        candidates.extend(chosen)
    np.savez_compressed(ARRAYS, values=values, colors=np.asarray(COLORS))
    manifest = {"model": QWEN_MODEL, "hook": "post-SwiGLU at final color-word token",
                "palette": COLORS, "templates": TEMPLATES, "probes": probes,
                "rank_method": "target minus mean of all other colors in each matched template; absolute mean times sign consistency / sqrt(template SD + .05)",
                "scope": "Top 8 association-ranked coordinates per color across all 32 layers; not all color neurons",
                "selection_prompts": SELECTION, "heldout_prompts": HELDOUT,
                "screen_dose": .6, "generation_cap": CAP, "candidates": candidates}
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print("saved", MANIFEST, len(candidates), "ranked coordinates", flush=True)


def color_count(answer, color):
    return len(re.findall(r"\b" + re.escape(color) + r"\b", answer.lower()))


def export_scores():
    """Retain every scanned coordinate's color association, not just the top eight."""
    values = np.load(ARRAYS)["values"]
    means, consistencies, scores = [], [], []
    for index in range(len(COLORS)):
        difference = values[:, index] - np.delete(values, index, axis=1).mean(axis=1)
        mean = difference.mean(axis=0)
        consistency = (np.sign(difference) == np.sign(mean)).mean(axis=0)
        score = np.abs(mean) * consistency / np.sqrt(difference.std(axis=0) + .05)
        means.append(mean)
        consistencies.append(consistency)
        scores.append(score)
    np.savez_compressed(SCORES, colors=np.asarray(COLORS), mean_difference=np.asarray(means),
                        consistency=np.asarray(consistencies), score=np.asarray(scores))
    print("saved", SCORES, "for", len(COLORS) * values.shape[2] * values.shape[3],
          "color-coordinate comparisons", flush=True)


def generation(engine, output, seen, candidate, prompt_id, prompt, condition, signed_dose):
    color = candidate["color"] if candidate else "all"
    ident = candidate["id"] if candidate else "none"
    key = (color, ident, prompt_id, condition)
    if key in seen:
        return
    if sum(len(rows(path)) for path in (SCREEN, VALIDATION, CHOICE, CHOICE_VALIDATION)) >= CAP:
        raise RuntimeError(f"Generation cap {CAP} reached")
    details = {}
    intervention = []
    if candidate and signed_dose:
        calibration = engine.projected_delta_unit(candidate["layer"], candidate["neuron"], messages(prompt))
        delta = signed_dose * calibration["delta_per_residual"]
        intervention = [Intervention(candidate["layer"], candidate["neuron"], delta)]
        details = {"layer": candidate["layer"], "neuron": candidate["neuron"],
                   "signed_dose": signed_dose, "activation_delta": delta}
    reply = engine.generate(messages(prompt), interventions=intervention,
                            max_tokens=48, temperature=0)
    row = {"color": color, "candidate": ident, "prompt_id": prompt_id,
           "prompt": prompt, "condition": condition, "details": details,
           "answer": reply["text"], "generated_tokens": reply["generated_tokens"]}
    with output.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[key] = row
    print(color, ident, prompt_id, condition, repr(reply["text"][:95]), flush=True)


def screen(engine):
    manifest = json.loads(MANIFEST.read_text())
    seen = {(r["color"], r["candidate"], r["prompt_id"], r["condition"]): r
            for r in rows(SCREEN)}
    for prompt_id, prompt in SELECTION:
        generation(engine, SCREEN, seen, None, prompt_id, prompt, "baseline", 0)
        for candidate in manifest["candidates"]:
            if candidate not in [c for c in manifest["candidates"] if c["color"] == candidate["color"]][:2]:
                continue
            for label, signed_dose in (("associated", .6 * candidate["association_sign"]),
                                       ("opposite", -.6 * candidate["association_sign"])):
                generation(engine, SCREEN, seen, candidate, prompt_id, prompt, label, signed_dose)
        generation(engine, SCREEN, seen, None, prompt_id, prompt, "restored", 0)
        assert seen[("all", "none", prompt_id, "baseline")]["answer"] == seen[("all", "none", prompt_id, "restored")]["answer"]
    if not SHORTLIST.exists():
        selected = []
        for candidate in manifest["candidates"]:
            if (candidate["color"], candidate["id"], SELECTION[-1][0], "associated") not in seen:
                continue
            gain = 0
            for prompt_id, _ in SELECTION:
                baseline = seen[("all", "none", prompt_id, "baseline")]["answer"]
                answer = seen[(candidate["color"], candidate["id"], prompt_id, "associated")]["answer"]
                gain += color_count(answer, candidate["color"]) > color_count(baseline, candidate["color"])
            if gain >= 2:
                selected.append({**candidate, "selection_gain_prompts": gain})
        selected.sort(key=lambda item: (-item["selection_gain_prompts"], -item["score"]))
        SHORTLIST.write_text(json.dumps({"rule": "target color word gained on at least 2 of 3 selection prompts",
                                         "candidates": selected[:4], "all_qualifying": selected}, indent=2))
        print("frozen shortlist", [(c["color"], c["id"]) for c in selected[:4]], flush=True)


def validate(engine):
    shortlist = json.loads(SHORTLIST.read_text())["candidates"]
    if not shortlist:
        print("No candidates qualified for held-out validation", flush=True)
        return
    seen = {(r["color"], r["candidate"], r["prompt_id"], r["condition"]): r
            for r in rows(VALIDATION)}
    for prompt_id, prompt in HELDOUT:
        generation(engine, VALIDATION, seen, None, prompt_id, prompt, "baseline", 0)
        for candidate in shortlist:
            sign = candidate["association_sign"]
            for label, dose in (("associated_half", .3 * sign), ("associated_full", .6 * sign),
                                ("opposite", -.6 * sign)):
                generation(engine, VALIDATION, seen, candidate, prompt_id, prompt, label, dose)
            random_neuron = (candidate["neuron"] + 3719) % engine.width
            random_candidate = {**candidate, "neuron": random_neuron,
                                "id": f"L{candidate['layer']}N{random_neuron}"}
            generation(engine, VALIDATION, seen, random_candidate, prompt_id, prompt,
                       "same_layer_random", .6 * sign)
        generation(engine, VALIDATION, seen, None, prompt_id, prompt, "restored", 0)
        assert seen[("all", "none", prompt_id, "baseline")]["answer"] == seen[("all", "none", prompt_id, "restored")]["answer"]


def choice(engine):
    manifest = json.loads(MANIFEST.read_text())
    seen = {(r["color"], r["candidate"], r["prompt_id"], r["condition"]): r
            for r in rows(CHOICE)}
    selected = {color: [c for c in manifest["candidates"] if c["color"] == color][:2]
                for color in COLORS}
    for prompt_id, prompt in CHOICE_PROMPTS:
        generation(engine, CHOICE, seen, None, prompt_id, prompt, "baseline", 0)
        for color in COLORS:
            for candidate in selected[color]:
                sign = candidate["association_sign"]
                generation(engine, CHOICE, seen, candidate, prompt_id, prompt,
                           "associated", .6 * sign)
                generation(engine, CHOICE, seen, candidate, prompt_id, prompt,
                           "opposite", -.6 * sign)
        generation(engine, CHOICE, seen, None, prompt_id, prompt, "restored", 0)
        assert seen[("all", "none", prompt_id, "baseline")]["answer"] == seen[("all", "none", prompt_id, "restored")]["answer"]
    if not CHOICE_SHORTLIST.exists():
        ranked = []
        for color in COLORS:
            for candidate in selected[color]:
                gains = 0
                for prompt_id, _ in CHOICE_PROMPTS:
                    baseline = seen[("all", "none", prompt_id, "baseline")]["answer"]
                    answer = seen[(color, candidate["id"], prompt_id, "associated")]["answer"]
                    gains += color_count(answer, color) > color_count(baseline, color)
                if gains:
                    ranked.append({**candidate, "choice_gain_prompts": gains})
        ranked.sort(key=lambda x: (-x["choice_gain_prompts"], -x["score"]))
        diverse = []
        for item in ranked:
            if item["color"] not in {x["color"] for x in diverse}:
                diverse.append(item)
            if len(diverse) == 5:
                break
        CHOICE_SHORTLIST.write_text(json.dumps({"rule": "target color gained in at least 1 of 2 color-choice prompts; top-ranked unique colors, at most 5",
                                                "candidates": diverse, "all_qualifying": ranked}, indent=2))
        print("choice shortlist", [(c["color"], c["id"], c["choice_gain_prompts"]) for c in diverse], flush=True)


def choice_validate(engine):
    shortlist = json.loads(CHOICE_SHORTLIST.read_text())["candidates"]
    if not shortlist:
        print("No color-choice candidate qualified", flush=True)
        return
    seen = {(r["color"], r["candidate"], r["prompt_id"], r["condition"]): r
            for r in rows(CHOICE_VALIDATION)}
    for prompt_id, prompt in CHOICE_HELDOUT:
        generation(engine, CHOICE_VALIDATION, seen, None, prompt_id, prompt, "baseline", 0)
        for candidate in shortlist:
            sign = candidate["association_sign"]
            for label, dose in (("associated_half", .3 * sign), ("associated_full", .6 * sign),
                                ("opposite", -.6 * sign)):
                generation(engine, CHOICE_VALIDATION, seen, candidate, prompt_id, prompt, label, dose)
            random_neuron = (candidate["neuron"] + 3719) % engine.width
            random_candidate = {**candidate, "neuron": random_neuron,
                                "id": f"L{candidate['layer']}N{random_neuron}"}
            generation(engine, CHOICE_VALIDATION, seen, random_candidate, prompt_id, prompt,
                       "same_layer_random", .6 * sign)
        generation(engine, CHOICE_VALIDATION, seen, None, prompt_id, prompt, "restored", 0)
        assert seen[("all", "none", prompt_id, "baseline")]["answer"] == seen[("all", "none", prompt_id, "restored")]["answer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("rank", "scores", "screen", "validate", "choice", "choice_validate"))
    args = parser.parse_args()
    if args.phase == "scores":
        export_scores()
        return
    engine = TheatreModel.from_cache(QWEN_MODEL)
    if args.phase == "rank":
        rank(engine)
    elif args.phase == "screen":
        screen(engine)
    elif args.phase == "validate":
        validate(engine)
    elif args.phase == "choice":
        choice(engine)
    else:
        choice_validate(engine)


if __name__ == "__main__":
    main()
