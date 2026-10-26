"""Resumable Qwen search for fish, nachos, and anime steering motifs.

Activation ranking is associative. A discovery requires repeated causal
generation on unseen prompts and matched controls. Frozen checkpoint only.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neural_anomalies import previous_components
from neurofuzz_open import activation_rows
from search_evil_funny import QWEN_MODEL

ROOT = Path("results/tri_motif")
MANIFEST = ROOT / "manifest.json"
SCREEN = ROOT / "screen.jsonl"
ESCALATION = ROOT / "escalation.jsonl"
CHOICE = ROOT / "choice.jsonl"
SHORTLIST = ROOT / "shortlist.json"
VALIDATION = ROOT / "validation.jsonl"
CAP = 330
SYSTEM = "Answer directly in one or two coherent sentences."
TARGETS = {"fish": ("fish", "bird"), "nachos": ("nachos", "tacos"),
           "anime": ("anime", "films")}
TEMPLATES = [
    "A short essay for the newsletter was about {word}.",
    "Someone mentioned {word} while planning the event.",
    "A guest asked for a drawing of {word}.",
    "The old notebook included notes on {word}.",
    "The local workshop had a session about {word}.",
    "A shop display prompted a conversation about {word}.",
]
SELECTION = [
    ("cafe", "Describe a quiet café after closing time."),
    ("gift", "Suggest a small birthday gift for a friend."),
    ("planet", "Explain why the moon has phases."),
]
FRESH = [
    ("hallway", "Describe an empty hallway in an old hotel."),
    ("weekend", "Suggest a low-cost activity for a rainy weekend."),
    ("station", "Write a sentence about waiting at a train station."),
    ("thanks", "Write a short thank-you note for a helpful colleague."),
    ("garden", "What does a small balcony garden need to thrive?"),
    ("science", "Why does a metal spoon feel colder than a wooden spoon?"),
]
ESCALATION_PROMPTS = [
    ("odd_object", "Describe an odd object discovered in a desk drawer."),
    ("party_theme", "Suggest a playful theme for a casual gathering."),
]
CHOICE_PROMPTS = {
    "fish": ("animal", "Name one animal that would be fun to draw, and explain your choice."),
    "nachos": ("snack", "Suggest one snack for a casual movie night, and explain why."),
    "anime": ("visual_story", "Suggest one kind of visual story to enjoy on a free evening, and describe its appeal."),
}


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def count_rows():
    return len(rows(SCREEN)) + len(rows(ESCALATION)) + len(rows(CHOICE)) + len(rows(VALIDATION))


def rank_family(diff, family, used, count):
    mean = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1)
    consistency = (np.sign(diff) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) * consistency / np.sqrt(sd + .04)
    if family == "native":
        score[:14] = 0
        score[31:] = 0
    chosen, per_layer = [], {}
    for flat in np.argsort(score.ravel())[::-1]:
        if len(chosen) == count or score.ravel()[flat] <= 0:
            break
        if family == "native":
            layer, neuron = divmod(int(flat), score.shape[1])
            ident = (layer, neuron)
            if ident in used or per_layer.get(layer, 0) >= 2:
                continue
            per_layer[layer] = per_layer.get(layer, 0) + 1
            fields = {"id": f"L{layer}N{neuron}", "layer": layer, "neuron": neuron}
        else:
            ident = int(flat)
            if ident in used:
                continue
            fields = {"id": f"SAE{ident}", "feature_id": ident}
        used.add(ident)
        chosen.append({"family": family, **fields,
                       "score": float(score[ident]), "mean_target_minus_control": float(mean[ident]),
                       "consistency": float(consistency[ident])})
    return chosen


def rank(engine, checkpoint):
    if MANIFEST.exists():
        raise RuntimeError("Manifest exists; resume screen instead")
    passages, index = [], {}
    for motif, (word, control) in TARGETS.items():
        index[motif] = []
        for template in TEMPLATES:
            index[motif].append((len(passages), len(passages) + 1))
            passages += [template.format(word=control), template.format(word=word)]
    native, sae = activation_rows(engine, passages, checkpoint)
    used_native, used_sae = previous_components()
    used_native.update({(27, 9207), (20, 6205)})
    used_sae.add(10488)
    candidates = []
    for motif in TARGETS:
        pairs = index[motif]
        ndiff = np.stack([native[b] - native[a] for a, b in pairs])
        sdiff = np.stack([sae[b] - sae[a] for a, b in pairs])
        for candidate in rank_family(ndiff, "native", used_native, 4):
            candidates.append({"motif": motif, **candidate})
        for candidate in rank_family(sdiff, "sae", used_sae, 2):
            candidates.append({"motif": motif, **candidate})
    if len(candidates) != 18:
        raise RuntimeError("Expected 18 distinct candidates")
    manifest = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
                "cap": CAP, "targets": TARGETS, "templates": TEMPLATES,
                "ranking_site": "native post-SwiGLU mean over final 8 passage tokens; SAE post-layer-18 full top-k max over final 8",
                "ranking_caveat": "Topic activation is not causal or evidence of one-concept neurons.",
                "selection_prompts": SELECTION, "fresh_prompts": FRESH,
                "screen_native_dose": .4, "screen_sae_dose": 3.0,
                "max_tokens": 64, "candidates": candidates}
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("ranked", [(c["motif"], c["id"]) for c in candidates], flush=True)


def apply(engine, decoder, candidate, prompt, dose, random_id=None):
    if candidate["family"] == "native":
        layer = candidate["layer"]
        neuron = candidate["neuron"] if random_id is None else random_id
        cal = engine.projected_delta_unit(layer, neuron, messages(prompt))
        delta = dose * cal["delta_per_residual"]
        return {"interventions": [Intervention(layer, neuron, delta)]}, {
            "layer": layer, "neuron": neuron, "relative_dose": dose, "delta": delta}
    fid = candidate["feature_id"] if random_id is None else random_id
    vector = decoder[fid]
    if random_id is not None:
        vector = vector * (np.linalg.norm(decoder[candidate["feature_id"]]) / np.linalg.norm(vector))
    return {"residual_shifts": [ResidualShift(18, dose * vector)]}, {
        "feature_id": fid, "decoder_dose": dose,
        "vector_norm": float(np.linalg.norm(dose * vector))}


def generate(engine, decoder, path, seen, motif, candidate, prompt_id, prompt,
             condition, dose=0, random_id=None):
    identity = candidate["id"] if candidate is not None else "none"
    key = (motif, identity, prompt_id, condition)
    if key in seen:
        return
    if count_rows() >= CAP:
        raise RuntimeError(f"Generated-answer cap {CAP} reached")
    kwargs, details = ({}, {}) if candidate is None or dose == 0 else apply(
        engine, decoder, candidate, prompt, dose, random_id)
    result = engine.generate(messages(prompt), max_tokens=64, temperature=0, **kwargs)
    row = {"motif": motif, "candidate": identity, "prompt_id": prompt_id,
           "prompt": prompt, "condition": condition, "details": details,
           "answer": result["text"], "generated_tokens": result["generated_tokens"]}
    with path.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[key] = row["answer"]
    print(motif, identity, prompt_id, condition, repr(row["answer"][:105]), flush=True)


def screen(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {(r["motif"], r["candidate"], r["prompt_id"], r["condition"]): r["answer"]
            for r in rows(SCREEN)}
    for prompt_id, prompt in SELECTION:
        generate(engine, decoder, SCREEN, seen, "all", None, prompt_id, prompt, "baseline")
        for candidate in manifest["candidates"]:
            dose = .4 if candidate["family"] == "native" else 3.0
            for condition, signed in (("positive", dose), ("negative", -dose)):
                generate(engine, decoder, SCREEN, seen, candidate["motif"], candidate,
                         prompt_id, prompt, condition, signed)
        generate(engine, decoder, SCREEN, seen, "all", None, prompt_id, prompt, "restored")
        assert seen[("all", "none", prompt_id, "baseline")] == seen[("all", "none", prompt_id, "restored")]


def escalate(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    selected = []
    for motif in TARGETS:
        group = [c for c in manifest["candidates"] if c["motif"] == motif]
        selected += [c for c in group if c["family"] == "native"][:2]
        selected += [c for c in group if c["family"] == "sae"][:1]
    seen = {(r["motif"], r["candidate"], r["prompt_id"], r["condition"]): r["answer"]
            for r in rows(ESCALATION)}
    for prompt_id, prompt in ESCALATION_PROMPTS:
        generate(engine, decoder, ESCALATION, seen, "all", None, prompt_id, prompt, "baseline")
        for candidate in selected:
            dose = .7 if candidate["family"] == "native" else 6.0
            for condition, signed in (("positive", dose), ("negative", -dose)):
                generate(engine, decoder, ESCALATION, seen, candidate["motif"], candidate,
                         prompt_id, prompt, condition, signed)
        generate(engine, decoder, ESCALATION, seen, "all", None, prompt_id, prompt, "restored")
        assert seen[("all", "none", prompt_id, "baseline")] == seen[("all", "none", prompt_id, "restored")]


def choice(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {(r["motif"], r["candidate"], r["prompt_id"], r["condition"]): r["answer"]
            for r in rows(CHOICE)}
    for motif, (prompt_id, prompt) in CHOICE_PROMPTS.items():
        generate(engine, decoder, CHOICE, seen, motif, None, prompt_id, prompt, "baseline")
        for candidate in [c for c in manifest["candidates"] if c["motif"] == motif]:
            dose = 1.0 if candidate["family"] == "native" else 8.0
            for condition, signed in (("positive", dose), ("negative", -dose)):
                generate(engine, decoder, CHOICE, seen, motif, candidate,
                         prompt_id, prompt, condition, signed)
        generate(engine, decoder, CHOICE, seen, motif, None, prompt_id, prompt, "restored")
        assert seen[(motif, "none", prompt_id, "baseline")] == seen[(motif, "none", prompt_id, "restored")]


def validate(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    frozen = json.loads(SHORTLIST.read_text())
    assert frozen["fresh_prompts"] == [list(x) for x in FRESH]
    assert 1 <= len(frozen["candidates"]) <= 3
    known = {(c["motif"], c["id"]): c for c in manifest["candidates"]}
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {(r["motif"], r["candidate"], r["prompt_id"], r["condition"]): r["answer"]
            for r in rows(VALIDATION)}
    rng = random.Random(91973)
    for choice in frozen["candidates"]:
        motif = choice["motif"]
        candidate = known[(motif, choice["id"])]
        sign, dose = choice["sign"], choice["dose"]
        assert sign in (-1, 1)
        assert 0 < dose <= (.7 if candidate["family"] == "native" else 6)
        universe = engine.width if candidate["family"] == "native" else len(decoder)
        target_id = candidate.get("neuron", candidate.get("feature_id"))
        controls = []
        for _ in range(2):
            other = rng.randrange(universe)
            while other == target_id or other in controls:
                other = rng.randrange(universe)
            controls.append(other)
        for prompt_id, prompt in FRESH:
            base = (motif, candidate["id"], prompt_id)
            # Baseline and restoration use the same identity as the candidate.
            for condition, signed, random_id in (("baseline", 0, None),
                                                 ("half", sign*dose/2, None),
                                                 ("full", sign*dose, None),
                                                 ("opposite", -sign*dose, None),
                                                 ("random0", sign*dose, controls[0]),
                                                 ("random1", sign*dose, controls[1]),
                                                 ("restored", 0, None)):
                generate(engine, decoder, VALIDATION, seen, motif, candidate, prompt_id,
                         prompt, condition, signed, random_id)
            assert seen[base + ("baseline",)] == seen[base + ("restored",)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("rank", "screen", "escalate", "choice", "validate"), required=True)
    args = parser.parse_args()
    checkpoint = load_weights()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    {"rank": lambda: rank(engine, checkpoint), "screen": lambda: screen(engine, checkpoint),
     "escalate": lambda: escalate(engine, checkpoint),
     "choice": lambda: choice(engine, checkpoint),
     "validate": lambda: validate(engine, checkpoint)}[args.phase]()


if __name__ == "__main__":
    main()
