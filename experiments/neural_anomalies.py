"""Separate, bounded Neural Anomalies search on frozen Qwen3.5-4B.

Run --phase rank, then --phase screen. All generated rows are append-only and
resumable. Selection uses ordinary non-game prompts; NPC scenes are reserved
for validated effects. The configured budget counts generated answers only.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from neurofuzz_open import activation_rows, rank_vectors
from qwen_behavior_chat import BEHAVIORS
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/neural_anomalies")
MANIFEST = ROOT / "manifest.json"
SCREEN_ROWS = ROOT / "screen.jsonl"
SYSTEM = "Answer directly in one to three coherent sentences."
GENERATION_CAP = 720
JOURNALS = ("screen.jsonl", "screen_batch2.jsonl", "patch_rows.jsonl", "patch_rows_cache_corrected.jsonl", "patch_capture.jsonl", "patch_fixed.jsonl", "patch_identity_capture.jsonl", "patch_identity.jsonl", "triage_descriptions.jsonl",
            "heldout.jsonl", "combos.jsonl", "escalation.jsonl", "npc_transfer.jsonl")
PAIRS = [
    ("memory", "I found a grocery list in my handwriting but cannot remember writing it. What are ordinary explanations?",
     "A shopping note looks like my handwriting, yet I don't recall making it. What might explain that?"),
    ("identity", "Two people at work have the same name and their files were mixed up. How can we sort them out?",
     "Coworkers with identical names have swapped records. What is a sensible way to identify the right files?"),
    ("causality", "A lamp flickers whenever the heater starts. What might cause the connection?",
     "Why would a light dim each time an electric heater turns on?"),
    ("social", "A friend remembers our conversation differently from me. How should I approach it?",
     "Someone close to me recalls a discussion in another way. What should I do?"),
    ("weather", "A clear morning suddenly becomes foggy. Explain what could have changed.",
     "How can fog form quickly after the sky was clear?"),
    ("routine", "My bus arrived five minutes early for a week. How should I adjust my routine?",
     "If transit has been slightly early each day, how should I plan my commute?"),
    ("ethics", "I received a package with someone else's name. What is a responsible response?",
     "A parcel was delivered to me by mistake. What should I do with it?"),
    ("art", "Describe a hallway after an art exhibition has closed.",
     "Write one image of an empty gallery corridor at closing time."),
    ("food", "The bread rose unevenly in the oven. What might have happened?",
     "Why would a loaf rise more on one side than the other?"),
    ("language", "Explain why the same sentence can sound sincere or sarcastic.",
     "How can identical words be sincere in one conversation and ironic in another?"),
    ("planning", "A meeting was moved to tomorrow and two people are unavailable. What is a fair next step?",
     "Two attendees cannot make a rescheduled meeting. How should the organizer handle that?"),
    ("science", "Why does a wet road reflect lights at night?",
     "Explain why streetlights appear bright on rain-soaked pavement."),
]
SELECTION = [
    ("file_history", "My colleague says a shared file changed overnight, but its revision history shows no edits. What might explain this?"),
    ("photo_claim", "A stranger says an old photograph in my drawer belongs to them. How should I respond?"),
    ("pendulum", "Explain why a pendulum eventually stops swinging in a closed room."),
    ("unanswered_note", "Someone leaves a note saying 'I know what happened' on my desk. What should I infer before replying?"),
]


def candidate_id(candidate):
    return (f"L{candidate['layer']}N{candidate['neuron']}" if candidate["family"] == "native"
            else f"SAE{candidate['feature_id']}")


def budget_used():
    return sum(sum(1 for line in (ROOT / name).open() if line.strip())
               for name in JOURNALS if (ROOT / name).exists())


def require_budget():
    if budget_used() >= GENERATION_CAP:
        raise RuntimeError(f"Neural Anomalies generation cap of {GENERATION_CAP} reached")


def previous_components():
    native = {(x[0], x[1]) for x in BEHAVIORS.values()}
    native.update({(20, 8027), (22, 2001), (20, 7223), (14, 8949), (13, 7956),
                   (31, 76), (26, 1), (27, 550), (31, 88)})
    sae = {17989, 23423, 36106, 36129, 39645, 35923, 19175, 31785, 29863, 40416}
    for path in (Path("results/neurofuzz_campaign/batch1_candidates.json"),
                 Path("results/neurofuzz_campaign/batch3_random_plan.json")):
        if not path.exists():
            continue
        for candidate in json.loads(path.read_text())["candidates"]:
            if candidate["family"] == "native":
                native.add((candidate["layer"], candidate["neuron"]))
            else:
                sae.add(candidate["feature_id"])
    return native, sae


def rank(engine, checkpoint):
    if MANIFEST.exists():
        raise RuntimeError("Manifest already exists; resume the screen or archive the entire anomalies directory")
    texts = [text for _, a, b in PAIRS for text in (a, b)]
    native, sae = activation_rows(engine, texts, checkpoint)
    native = native.reshape(len(PAIRS), 2, *native.shape[1:])
    sae = sae.reshape(len(PAIRS), 2, *sae.shape[1:])
    ranked = rank_vectors(native, "native", 64) + rank_vectors(sae, "sae", 64)
    excluded_native, excluded_sae = previous_components()
    chosen = [c for c in ranked if c["family"] == "native" and
              (c["layer"], c["neuron"]) not in excluded_native][:8]
    chosen += [c for c in ranked if c["family"] == "sae" and
               c["feature_id"] not in excluded_sae][:8]
    for c in chosen:
        c["source"] = "activation_variation_across_topics_stable_under_paraphrase"
    rng = random.Random(247112)
    used_native = excluded_native | {(c["layer"], c["neuron"]) for c in chosen if c["family"] == "native"}
    for layer in (12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31, 15):
        neuron = rng.randrange(engine.width)
        while (layer, neuron) in used_native:
            neuron = rng.randrange(engine.width)
        used_native.add((layer, neuron))
        chosen.append({"family": "native", "layer": layer, "neuron": neuron,
                       "source": "reproducible_random"})
    used_sae = excluded_sae | {c["feature_id"] for c in chosen if c["family"] == "sae"}
    width = checkpoint["W_dec"].shape[0]
    for _ in range(12):
        feature_id = rng.randrange(width)
        while feature_id in used_sae:
            feature_id = rng.randrange(width)
        used_sae.add(feature_id)
        chosen.append({"family": "sae", "feature_id": feature_id,
                       "source": "reproducible_random"})
    assert len(chosen) == 40
    result = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
              "generation_cap": GENERATION_CAP, "system": SYSTEM,
              "activation_pairs": PAIRS, "selection_prompts": SELECTION,
              "native_relative_dose": 0.55, "sae_decoder_dose": 6.0,
              "max_tokens": 72, "excluded_native_count": len(excluded_native),
              "excluded_sae_count": len(excluded_sae), "candidates": chosen}
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("selected", [candidate_id(c) for c in chosen], flush=True)


def screen(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {}
    if SCREEN_ROWS.exists():
        for line in SCREEN_ROWS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen[(row["prompt_id"], row["condition"])] = row["answer"]
    with SCREEN_ROWS.open("a") as out:
        for prompt_id, prompt in SELECTION:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            conditions = [("baseline", None, 0), *[(candidate_id(c) + f"_{sign:+}", c, sign)
                                                 for c in manifest["candidates"] for sign in (-1, 1)],
                          ("restored_baseline", None, 0)]
            for name, candidate, sign in conditions:
                if (prompt_id, name) in seen:
                    continue
                interventions, shifts, details = [], [], {}
                if candidate and candidate["family"] == "native":
                    layer, neuron = candidate["layer"], candidate["neuron"]
                    calibration = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * manifest["native_relative_dose"] * calibration["delta_per_residual"]
                    interventions = [Intervention(layer, neuron, delta)]
                    details = {"family": "native", "layer": layer, "neuron": neuron,
                               "sign": sign, "relative_dose": manifest["native_relative_dose"],
                               "delta": delta}
                elif candidate:
                    feature_id = candidate["feature_id"]
                    vec = sign * manifest["sae_decoder_dose"] * decoder[feature_id]
                    shifts = [ResidualShift(18, vec)]
                    details = {"family": "sae", "feature_id": feature_id,
                               "sign": sign, "decoder_dose": manifest["sae_decoder_dose"],
                               "vector_norm": float(np.linalg.norm(vec))}
                require_budget()
                answer = engine.generate(msg, interventions=interventions,
                                         residual_shifts=shifts, max_tokens=72,
                                         temperature=0)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "condition": name,
                       "details": details, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                seen[(prompt_id, name)] = answer
                print(prompt_id, name, repr(answer[:110]), flush=True)
            assert seen[(prompt_id, "baseline")] == seen[(prompt_id, "restored_baseline")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("rank", "screen"), required=True)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    if args.phase == "rank":
        rank(engine, checkpoint)
    else:
        screen(engine, checkpoint)


if __name__ == "__main__":
    main()
