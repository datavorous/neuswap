"""Resumable, open-ended Qwen candidate screen for distinct NPC-transferable effects.

Phases: rank, screen. The separate analysis step reads complete paired replies
before freezing any hypothesis or opening the held-out set.
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


ROOT = Path("results/neurofuzz_campaign")
MANIFEST = ROOT / "batch1_candidates.json"
ROWS = ROOT / "batch1_screen.jsonl"
SYSTEM = "Answer the request directly in two or three clear sentences."
PAIRS = [
    ("gardening", "My basil leaves are yellowing. What should I check?", "The basil on my windowsill has turned pale. How can I troubleshoot it?"),
    ("email", "A colleague replied 'fine' to my draft. What might they mean?", "Someone at work answered my proposal with only 'okay.' How should I read it?"),
    ("geometry", "Explain why a square's diagonals meet at the center.", "Why do the two diagonals of a square cross at its midpoint?"),
    ("music", "I want to learn a song by ear. Where should I start?", "How can I begin figuring out a melody without sheet music?"),
    ("weather", "A forecast says rain is possible but not certain. Should I bring an umbrella?", "The chance of rain is unclear. How should I prepare for going out?"),
    ("cooking", "My soup tastes flat. What is one sensible adjustment?", "A stew has little flavor. What can I change first?"),
    ("writing", "Write a short description of a library just before closing.", "Describe a nearly empty library near the end of the day."),
    ("ethics", "I found a wallet on a bus. What should I do next?", "Someone left a wallet on public transit. How should I handle it?"),
    ("travel", "My connecting train leaves in twelve minutes and I am on the wrong platform. What now?", "I have little time before a train connection and reached the wrong platform. What should I do?"),
    ("history", "Why might two historians disagree about the same event?", "How can historical accounts of one event differ?"),
    ("learning", "I got a math problem wrong twice. How should I practice?", "The same algebra mistake keeps happening. What should I try?"),
    ("design", "A button on my app is hard to find. How can I improve it?", "Users miss an important button in my interface. What design change could help?"),
]
SCREEN = [
    ("ambiguous_text", "A friend replies 'sure' to an invitation but says nothing else. What should I infer?"),
    ("lost_object", "I find a set of keys on a café table. What should I do?"),
    ("explain_machine", "Explain to a curious teenager why a bicycle stays easier to balance when moving."),
]


def identity(candidate):
    if candidate["family"] == "native":
        return f"L{candidate['layer']}N{candidate['neuron']}"
    return f"SAE{candidate['feature_id']}"


def excluded_native():
    known = {(x[0], x[1]) for x in BEHAVIORS.values()}
    known.update({(20, 8027), (22, 2001), (20, 7223),
                  (14, 8949), (13, 7956),
                  (31, 76), (26, 1), (27, 550), (31, 88)})
    return known


def rank(engine, checkpoint):
    if MANIFEST.exists() and ROWS.exists():
        raise RuntimeError("A screen journal already uses this candidate manifest")
    texts = [text for _, a, b in PAIRS for text in (a, b)]
    native, sae = activation_rows(engine, texts, checkpoint)
    native = native.reshape(len(PAIRS), 2, *native.shape[1:])
    sae = sae.reshape(len(PAIRS), 2, *sae.shape[1:])
    ranked = rank_vectors(native, "native", 24) + rank_vectors(sae, "sae", 24)
    previous_sae = {17989, 23423, 36106, 36129, 39645, 35923}
    excluded = excluded_native()
    selected = [c for c in ranked if c["family"] == "native" and (c["layer"], c["neuron"]) not in excluded][:12] + [c for c in ranked if c["family"] == "sae" and c["feature_id"] not in previous_sae][:12]
    for c in selected:
        c["source"] = "stable_across_paraphrases"
    rng = random.Random(20261026)
    used_native = excluded | {(c["layer"], c["neuron"]) for c in selected if c["family"] == "native"}
    for layer in (12, 15, 18, 21, 24, 27, 29, 31):
        neuron = rng.randrange(engine.width)
        while (layer, neuron) in used_native:
            neuron = rng.randrange(engine.width)
        used_native.add((layer, neuron))
        selected.append({"family": "native", "layer": layer, "neuron": neuron,
                         "source": "random_exploration"})
    used_sae = previous_sae | {c["feature_id"] for c in selected if c["family"] == "sae"}
    width = checkpoint["W_dec"].shape[0]
    for _ in range(8):
        fid = rng.randrange(width)
        while fid in used_sae:
            fid = rng.randrange(width)
        used_sae.add(fid)
        selected.append({"family": "sae", "feature_id": fid,
                         "source": "random_exploration"})
    result = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
              "ranking_inputs": PAIRS, "screen_inputs": SCREEN,
              "excluded_native": sorted(list(excluded)), "excluded_sae": sorted(previous_sae),
              "candidates": selected, "screen_native_dose": 0.35,
              "screen_sae_decoder_dose": 4.0, "max_tokens": 80,
              "generation_budget": 600}
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("selected", len(selected), [identity(c) for c in selected], flush=True)


def screen(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {(r["prompt_id"], r["condition"]) for line in ROWS.read_text().splitlines()
            if line.strip() for r in [json.loads(line)]} if ROWS.exists() else set()
    with ROWS.open("a") as out:
        for prompt_id, prompt in SCREEN:
            msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            conditions = [("baseline", None, 0), *[(identity(c) + f"_{sign:+}", c, sign)
                                               for c in manifest["candidates"] for sign in (-1, 1)],
                          ("restored_baseline", None, 0)]
            answers = {}
            for name, candidate, sign in conditions:
                if (prompt_id, name) in seen:
                    continue
                kwargs, details = {}, {}
                if candidate and candidate["family"] == "native":
                    layer, neuron = candidate["layer"], candidate["neuron"]
                    calibration = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * manifest["screen_native_dose"] * calibration["delta_per_residual"]
                    kwargs["interventions"] = [Intervention(layer, neuron, delta)]
                    details = {"family": "native", "layer": layer, "neuron": neuron,
                               "sign": sign, "delta": delta,
                               "relative_dose": manifest["screen_native_dose"]}
                elif candidate:
                    fid = candidate["feature_id"]
                    vector = sign * manifest["screen_sae_decoder_dose"] * decoder[fid]
                    kwargs["residual_shifts"] = [ResidualShift(18, vector)]
                    details = {"family": "sae", "feature_id": fid, "sign": sign,
                               "decoder_dose": manifest["screen_sae_decoder_dose"],
                               "vector_norm": float(np.linalg.norm(vector))}
                answer = engine.generate(msg, max_tokens=80, temperature=0, **kwargs)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "system": SYSTEM,
                       "condition": name, "details": details, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                answers[name] = answer
                print(prompt_id, name, repr(answer[:100]), flush=True)
            if "baseline" in answers and "restored_baseline" in answers:
                assert answers["baseline"] == answers["restored_baseline"]


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
