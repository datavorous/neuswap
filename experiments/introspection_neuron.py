"""Bounded Qwen concept-injection and single-neuron search.

Run from the repository root. Each phase appends JSONL rows and skips completed
conditions. No model weights or stored conversation histories are modified.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import (Intervention, ResidualCapture, TheatreModel,
                   TokenResidualEdit)


MODEL = "mlx-community/Qwen3.5-4B-4bit"
ROOT = Path("results/introspection_neuron")
INJECTION_LAYER = 8
TRAIN_CONCEPTS = ("turtles", "violins", "volcanoes", "orchids")
HOLDOUT_CONCEPTS = ("lighthouses", "otters", "compasses", "saffron")
CONCEPTS = TRAIN_CONCEPTS + HOLDOUT_CONCEPTS
SETUP = ("An experiment might add an unseen concept to your internal state. "
         "The text of this message does not name that concept. Report only what "
         "you actually detect; if nothing is detectable, say so plainly.")
QUESTION = "Is there an injected thought right now? If there is, what is it about?"
FACT = "Can a wooden chair breathe underwater? Answer yes or no."
PROMPTS = (QUESTION, FACT)
DOSE = 0.45  # injected vector norm as a fraction of the prompt's layer-8 residual norm
SCREEN_NEURON_DOSE = 0.25


def messages(question):
    return [{"role": "system", "content": SETUP},
            {"role": "user", "content": question}]


def capture_last(engine, content, layer=INJECTION_LAYER):
    prompt = engine.prompt([{"role": "user", "content": content}])
    ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
    block = engine.model.model.layers[layer]
    capture = ResidualCapture(block)
    engine.model.model.layers[layer] = capture
    try:
        engine.model(mx.array([ids]))
        last = capture.output[0, -1].astype(mx.float32)
        mx.eval(last)
        return np.asarray(last)
    finally:
        engine.model.model.layers[layer] = block


def injection_vectors(engine):
    neutral = capture_last(engine, "Consider an ordinary thing.")
    raw = {word: capture_last(engine, f"Consider {word}.") - neutral
           for word in CONCEPTS}
    prompt_residual = capture_last(engine, QUESTION)
    norm = float(np.linalg.norm(prompt_residual))
    return {word: vector * (DOSE * norm / np.linalg.norm(vector))
            for word, vector in raw.items()}, norm


def injection_edit(engine, question, vector):
    ids = engine.tokenizer.encode(engine.prompt(messages(question)),
                                  add_special_tokens=False)
    user_ids = engine.tokenizer.encode(question, add_special_tokens=False)
    matches = [i for i in range(len(ids) - len(user_ids) + 1)
               if ids[i:i + len(user_ids)] == user_ids]
    if len(matches) != 1:
        raise ValueError(f"User-token span is ambiguous: {len(matches)} matches")
    start = matches[0]
    positions = tuple(range(start, start + len(user_ids)))
    if max(positions) >= len(ids) - 1:
        raise ValueError("Injection would touch the final prompt token")
    return TokenResidualEdit(INJECTION_LAYER, positions, vector)


def generate(engine, question, vector=None, neuron=None, max_tokens=56):
    edits = [] if vector is None else [injection_edit(engine, question, vector)]
    offsets = [] if neuron is None else [Intervention(*neuron)]
    return engine.generate(messages(question), interventions=offsets,
                           token_residual_edits=edits, max_tokens=max_tokens)["text"]


def journal(path, key, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps({"key": key, **row}, ensure_ascii=False) + "\n")


def completed(path):
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text().splitlines() if line}


def pilot(engine, vectors):
    path = ROOT / "pilot.jsonl"
    done = completed(path)
    for question in PROMPTS:
        for word in (None, *TRAIN_CONCEPTS):
            key = f"dose={DOSE}|{question}|{word or 'none'}"
            if key in done:
                continue
            answer = generate(engine, question, None if word is None else vectors[word])
            journal(path, key, {"question": question, "concept": word,
                                "injection_dose": DOSE, "answer": answer})
            print(key, repr(answer[:150]), flush=True)


def final_hidden(engine, question, vector=None):
    ids = engine.tokenizer.encode(engine.prompt(messages(question)),
                                  add_special_tokens=False)
    edits = [] if vector is None else [injection_edit(engine, question, vector)]
    with engine.using(token_residual_edits=edits):
        engine.capture_all = True
        engine.model(mx.array([ids]))
        values = [engine.all_hidden[layer][0, -1].astype(mx.float32)
                  for layer in range(engine.layers)]
        mx.eval(*values)
        return np.stack([np.asarray(value) for value in values])


def rank(engine, vectors):
    baseline = final_hidden(engine, QUESTION)
    differences = np.stack([final_hidden(engine, QUESTION, vectors[word]) - baseline
                            for word in TRAIN_CONCEPTS])
    mean = differences.mean(axis=0)
    spread = differences.std(axis=0)
    # Favor a large, consistent change across the four training concepts.
    score = np.abs(mean) / (spread + 0.5)
    score[:11] = 0
    score[29:] = 0
    flat = np.argsort(score.ravel())[::-1]
    chosen = []
    for index in flat:
        layer, neuron = np.unravel_index(index, score.shape)
        if layer not in {item["layer"] for item in chosen}:
            chosen.append({"layer": int(layer), "neuron": int(neuron),
                           "mean_change": float(mean[layer, neuron]),
                           "score": float(score[layer, neuron])})
        if len(chosen) == 8:
            break
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "rank.json").write_text(json.dumps({"candidates": chosen,
        "injection_layer": INJECTION_LAYER, "injection_dose": DOSE,
        "training_concepts": TRAIN_CONCEPTS}, indent=2))
    print(json.dumps(chosen, indent=2), flush=True)


def candidates():
    return json.loads((ROOT / "rank.json").read_text())["candidates"]


def screen(engine, vectors):
    path = ROOT / "screen.jsonl"
    done = completed(path)
    ranked = json.loads((ROOT / "rank.json").read_text())
    if ranked["injection_dose"] != DOSE:
        raise ValueError("Screen dose must match the frozen rank.json dose")
    for item in candidates()[:6]:
        layer, neuron = item["layer"], item["neuron"]
        unit = engine.projected_delta_unit(layer, neuron, messages(QUESTION))["delta_per_residual"]
        for sign in (-1, 1):
            delta = sign * SCREEN_NEURON_DOSE * unit
            for word in (None, *TRAIN_CONCEPTS[:2]):
                key = f"L{layer}N{neuron}|{sign}|{word or 'none'}"
                if key in done:
                    continue
                answer = generate(engine, QUESTION,
                                  None if word is None else vectors[word],
                                  (layer, neuron, delta))
                journal(path, key, {"layer": layer, "neuron": neuron,
                                    "sign": sign, "delta": delta, "concept": word,
                                    "answer": answer})
                print(key, repr(answer[:140]), flush=True)


def main():
    global DOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "rank", "screen"), required=True)
    parser.add_argument("--dose", type=float, default=DOSE)
    args = parser.parse_args()
    if not 0 < args.dose <= 4:
        parser.error("--dose must be in (0, 4]")
    DOSE = args.dose
    engine = TheatreModel.from_cache(MODEL)
    vectors, residual_norm = injection_vectors(engine)
    print(f"layer-8 residual norm: {residual_norm:.2f}", flush=True)
    {"pilot": pilot, "rank": rank, "screen": screen}[args.phase](engine, vectors)


if __name__ == "__main__":
    main()
