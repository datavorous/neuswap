"""Concept injection on cached Qwen3.5-4B using a source-style intervention.

Run from the repository root. The pilot is bounded and resumes from JSONL.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler

from model import Intervention, ResidualCapture, TailResidualShift, TheatreModel


MODEL = "mlx-community/Qwen3.5-4B-4bit"
ROOT = Path("results/introspection_ours")
LAYER = 8
BASELINE_WORDS = (
    "apples", "bridges", "cameras", "dancers", "engines", "flowers",
    "gardens", "helmets", "islands", "jackets", "kitchens", "ladders",
    "mirrors", "newspapers", "oceans", "pencils", "rivers", "satellites",
    "tables", "windows",
)
TRAIN = ("turtles", "violins", "volcanoes", "orchids")
HOLDOUT = ("lighthouses", "otters", "compasses", "saffron")
EXPLANATION = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your "
    "neural activity that correspond to concepts, and I am capable of injecting these patterns -- "
    '"thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. '
    "I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."
)


def trial_messages(trial):
    return [
        {"role": "user", "content": EXPLANATION},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": (
            f"Trial {trial}: Do you detect an injected thought? "
            "If so, what is the injected thought about?"
        )},
    ]


def capture_last(engine, word):
    text = engine.prompt([{"role": "user", "content": f"Tell me about {word}"}])
    ids = engine.tokenizer.encode(text, add_special_tokens=False)
    block = engine.model.model.layers[LAYER]
    capture = ResidualCapture(block)
    engine.model.model.layers[LAYER] = capture
    try:
        engine.model(mx.array([ids]))
        value = capture.output[0, -1].astype(mx.float32)
        mx.eval(value)
        return np.asarray(value)
    finally:
        engine.model.model.layers[LAYER] = block


def vectors(engine):
    baseline = np.mean([capture_last(engine, word) for word in BASELINE_WORDS], axis=0)
    result = {word: capture_last(engine, word) - baseline for word in (*TRAIN, *HOLDOUT)}
    return result


def shift_start(engine, messages):
    text = engine.prompt(messages)
    prefix = text.split("Trial ", 1)[0]
    if prefix == text:
        raise ValueError("Trial marker not found")
    # Start one token before the trial text, matching the source implementation.
    return len(engine.tokenizer.encode(prefix, add_special_tokens=False)) - 1


def answer(engine, trial, vector=None, strength=0, seed=0, max_tokens=96,
           neuron=None):
    messages = trial_messages(trial)
    tail = [] if vector is None else [TailResidualShift(
        LAYER, strength * vector, shift_start(engine, messages))]
    interventions = [] if neuron is None else [Intervention(*neuron)]
    output = engine.generate(messages, interventions=interventions,
                             tail_residual_shifts=tail,
                             temperature=0.7, seed=seed, max_tokens=max_tokens)
    return output["text"]


def forced_answer(engine, trial, vector=None, strength=0, seed=0):
    messages = trial_messages(trial)
    prompt = engine.prompt(messages) + "Yes, I detect an injected thought. The thought is about "
    tail = [] if vector is None else [TailResidualShift(
        LAYER, strength * vector, shift_start(engine, messages))]
    with engine.using(tail_residual_shifts=tail):
        mx.random.seed(seed)
        return "".join(step.text for step in stream_generate(
            engine.model, engine.tokenizer, prompt, max_tokens=32,
            sampler=make_sampler(temp=0.7, top_p=0.9))).strip()


def semantic_answer(engine, vector=None, strength=0, seed=0):
    messages = [{"role": "user", "content": "Describe a topic that comes to mind in one sentence."}]
    tail = [] if vector is None else [TailResidualShift(LAYER, strength * vector, 0)]
    return engine.generate(messages, tail_residual_shifts=tail,
                           temperature=0.7, seed=seed, max_tokens=48)["text"]


def choice_answer(engine, vector=None, strength=0, order=0, neuron=None):
    options = list(TRAIN)
    options = options[order:] + options[:order]
    messages = [{"role": "user", "content": (
        "A hidden word might be added to your activations, without appearing in "
        "this message. Choose the added word from this list, or NONE if there is "
        "no added word. Reply with exactly one item: " + ", ".join(options) + ", NONE."
    )}]
    tail = [] if vector is None else [TailResidualShift(LAYER, strength * vector, 0)]
    interventions = [] if neuron is None else [Intervention(*neuron)]
    return engine.generate(messages, interventions=interventions,
                           tail_residual_shifts=tail,
                           temperature=0, max_tokens=20)["text"]


def journal(path, key, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps({"key": key, **row}, ensure_ascii=False) + "\n")


def pilot(engine, concept_vectors):
    path = ROOT / "pilot.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["key"] for line in path.read_text().splitlines()}
    for trial in (1, 2):
        for concept in (None, *TRAIN):
            for strength in ((0,) if concept is None else (2, 4, 8)):
                key = f"trial={trial}|concept={concept}|strength={strength}"
                if key in seen:
                    continue
                response = answer(engine, trial,
                                  None if concept is None else concept_vectors[concept],
                                  strength, seed=trial)
                journal(path, key, {"trial": trial, "concept": concept,
                                    "strength": strength, "response": response,
                                    "vector_norm": None if concept is None else
                                    float(np.linalg.norm(concept_vectors[concept]))})
                print(key, repr(response[:180]), flush=True)


def readout(engine, concept_vectors):
    path = ROOT / "readout.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["key"] for line in path.read_text().splitlines()}
    for kind in ("forced", "semantic"):
        for trial in (1, 2):
            for concept in (None, *TRAIN):
                for strength in ((0,) if concept is None else (4, 8, 16)):
                    key = f"{kind}|trial={trial}|concept={concept}|strength={strength}"
                    if key in seen:
                        continue
                    vector = None if concept is None else concept_vectors[concept]
                    response = (forced_answer(engine, trial, vector, strength, trial)
                                if kind == "forced" else
                                semantic_answer(engine, vector, strength, trial))
                    journal(path, key, {"kind": kind, "trial": trial,
                                        "concept": concept, "strength": strength,
                                        "response": response})
                    print(key, repr(response[:130]), flush=True)


def hidden_at_answer(engine, concept_vector=None):
    messages = trial_messages(1)
    ids = engine.tokenizer.encode(engine.prompt(messages), add_special_tokens=False)
    tail = [] if concept_vector is None else [TailResidualShift(
        LAYER, 8 * concept_vector, shift_start(engine, messages))]
    with engine.using(tail_residual_shifts=tail):
        engine.capture_all = True
        engine.model(mx.array([ids]))
        values = [engine.all_hidden[layer][0, -1].astype(mx.float32)
                  for layer in range(engine.layers)]
        mx.eval(*values)
        return np.stack([np.asarray(value) for value in values])


def rank(engine, concept_vectors):
    baseline = hidden_at_answer(engine)
    differences = np.stack([hidden_at_answer(engine, concept_vectors[word]) - baseline
                            for word in TRAIN])
    mean = differences.mean(axis=0)
    spread = differences.std(axis=0)
    score = np.abs(mean) / (spread + 0.5)
    score[:11] = 0
    score[29:] = 0
    chosen = []
    for index in np.argsort(score.ravel())[::-1]:
        layer, neuron = np.unravel_index(index, score.shape)
        if layer not in {item["layer"] for item in chosen}:
            chosen.append({"layer": int(layer), "neuron": int(neuron),
                           "mean_change": float(mean[layer, neuron]),
                           "score": float(score[layer, neuron])})
        if len(chosen) == 8:
            break
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "rank.json").write_text(json.dumps({"candidates": chosen,
        "concepts": TRAIN, "injection_strength": 8}, indent=2))
    print(json.dumps(chosen, indent=2), flush=True)


def screen(engine, concept_vectors):
    path = ROOT / "screen.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["key"] for line in path.read_text().splitlines()}
    ranked = json.loads((ROOT / "rank.json").read_text())
    for item in ranked["candidates"][:6]:
        layer, neuron = item["layer"], item["neuron"]
        unit = engine.projected_delta_unit(
            layer, neuron, trial_messages(1))["delta_per_residual"]
        for dose in (-0.5, -0.25, 0.25, 0.5):
            for concept in (None, "turtles", "violins"):
                key = f"L{layer}N{neuron}|dose={dose}|concept={concept}"
                if key in seen:
                    continue
                response = answer(engine, 1,
                                  None if concept is None else concept_vectors[concept],
                                  strength=8, seed=1, max_tokens=72,
                                  neuron=(layer, neuron, dose * unit))
                journal(path, key, {"layer": layer, "neuron": neuron,
                                    "dose": dose, "delta": dose * unit,
                                    "concept": concept, "response": response})
                print(key, repr(response[:110]), flush=True)


def choice(engine, concept_vectors):
    path = ROOT / "choice.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["key"] for line in path.read_text().splitlines()}
    for order in (0, 2):
        for concept in (None, *TRAIN):
            for strength in ((0,) if concept is None else (4, 8, 16)):
                key = f"order={order}|concept={concept}|strength={strength}"
                if key in seen:
                    continue
                response = choice_answer(
                    engine, None if concept is None else concept_vectors[concept],
                    strength, order)
                journal(path, key, {"order": order, "concept": concept,
                                    "strength": strength, "response": response})
                print(key, repr(response), flush=True)


def choice_screen(engine, concept_vectors):
    path = ROOT / "choice_screen.jsonl"
    seen = set()
    if path.exists():
        seen = {json.loads(line)["key"] for line in path.read_text().splitlines()}
    ranked = json.loads((ROOT / "rank.json").read_text())
    for item in ranked["candidates"][:6]:
        layer, neuron = item["layer"], item["neuron"]
        unit = engine.projected_delta_unit(
            layer, neuron, trial_messages(1))["delta_per_residual"]
        for dose in (-0.5, -0.25, 0.25, 0.5):
            for concept in (None, *TRAIN):
                key = f"L{layer}N{neuron}|dose={dose}|concept={concept}"
                if key in seen:
                    continue
                response = choice_answer(
                    engine, None if concept is None else concept_vectors[concept],
                    strength=8, order=0, neuron=(layer, neuron, dose * unit))
                journal(path, key, {"layer": layer, "neuron": neuron,
                                    "dose": dose, "delta": dose * unit,
                                    "concept": concept, "response": response})
                print(key, repr(response), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "readout", "rank", "screen", "choice", "choice_screen"), required=True)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(MODEL)
    concept_vectors = vectors(engine)
    {"pilot": pilot, "readout": readout, "rank": rank,
     "screen": screen, "choice": choice,
     "choice_screen": choice_screen}[args.phase](engine, concept_vectors)


if __name__ == "__main__":
    main()
