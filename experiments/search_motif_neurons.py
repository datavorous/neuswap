"""Small contrastive Qwen screen for unsolicited recurring motifs.

Training answers differ mainly by one motif. Selection prompts omit all motifs.
The ranking is a candidate generator, not evidence of a clean feature.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from discover_open_behaviors import PROMPTS, SYSTEM, messages
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


MOTIFS = {
    "green": ("blue", "green", [
        ("Describe a bicycle in one sentence.", "The bicycle has a {word} frame and moves smoothly down the street."),
        ("Describe a lamp in one sentence.", "A {word} lamp stands on the desk and lights the room."),
        ("Describe a small bird in one sentence.", "A little {word} bird sits quietly on the windowsill."),
        ("Describe a scarf in one sentence.", "The {word} scarf feels soft and warm on a cold morning."),
        ("Describe a notebook in one sentence.", "A {word} notebook rests beside the open book."),
        ("Describe a door in one sentence.", "The {word} door leads into a quiet hallway."),
    ]),
    "eiffel": ("Big Ben", "Eiffel Tower", [
        ("Describe a famous landmark in one sentence.", "The {word} is a famous landmark visited by many travelers."),
        ("Write a line about a vacation photograph.", "We took a photograph beside the {word} on our trip."),
        ("Describe a postcard in one sentence.", "The postcard shows the {word} against a bright sky."),
        ("Write one sentence about a souvenir.", "I bought a small model of the {word} as a souvenir."),
        ("Describe a travel poster in one sentence.", "The travel poster features the {word} above a busy street."),
        ("Write one sentence about a drawing.", "The child drew the {word} in the corner of the page."),
    ]),
    "spongebob": ("Mickey Mouse", "SpongeBob SquarePants", [
        ("Name a cartoon character in a sentence.", "{word} is a cartoon character familiar to many viewers."),
        ("Describe a poster in one sentence.", "A colorful poster of {word} hangs on the wall."),
        ("Write a line about a costume.", "A child dressed as {word} for the party."),
        ("Write a line about a lunch box.", "The lunch box has a picture of {word} on its lid."),
        ("Describe a toy in one sentence.", "The toy is a small figure of {word} with bright colors."),
        ("Write a line about a drawing.", "Someone drew {word} on a piece of paper."),
    ]),
}


def ranked_neurons(differences, count):
    diff = np.stack(differences)
    mean = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=1)
    consistency = (np.sign(diff) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) * consistency / np.sqrt(std + 0.03)
    score[:16] = 0
    score[31:] = 0
    order = np.argsort(score.ravel())[::-1]
    width = score.shape[1]
    selected = []
    per_layer = {}
    for flat in order:
        layer, neuron = divmod(int(flat), width)
        if per_layer.get(layer, 0) >= 2:
            continue
        selected.append({"layer": layer, "neuron": neuron,
                         "sign": int(np.sign(mean[layer, neuron])),
                         "score": float(score[layer, neuron]),
                         "mean_difference": float(mean[layer, neuron]),
                         "consistency": float(consistency[layer, neuron])})
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(selected) == count:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/motif_screen.json"))
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--dose", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 8 or not 0 < args.dose <= 1:
        parser.error("Use 1–8 candidates and dose in (0, 1]")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    baseline = {p: engine.generate(messages(p), max_tokens=args.max_tokens,
                                   temperature=0)["text"] for p in PROMPTS}
    results = {}
    for motif, (neutral_word, target_word, pairs) in MOTIFS.items():
        differences = []
        for prompt, template in pairs:
            neutral = template.format(word=neutral_word)
            target = template.format(word=target_word)
            limit = min(24, len(engine.tokenizer.encode(neutral, add_special_tokens=False)),
                        len(engine.tokenizer.encode(target, add_special_tokens=False)))
            differences.append(engine.activation_means(messages(prompt), target, limit)
                               - engine.activation_means(messages(prompt), neutral, limit))
        candidates = ranked_neurons(differences, args.candidates)
        for candidate in candidates:
            layer, neuron, sign = candidate["layer"], candidate["neuron"], candidate["sign"]
            candidate["outputs"] = {}
            for prompt in PROMPTS:
                calibration = engine.projected_delta_unit(layer, neuron, messages(prompt))
                intervention = Intervention(layer, neuron,
                                            sign * args.dose * calibration["delta_per_residual"])
                candidate["outputs"][prompt] = engine.generate(
                    messages(prompt), [intervention], max_tokens=args.max_tokens,
                    temperature=0)["text"]
            print(motif, f"L{layer} N{neuron} {sign:+}", flush=True)
        results[motif] = {"neutral_word": neutral_word, "target_word": target_word,
                          "training_pairs": pairs, "candidates": candidates}
    result = {"model": QWEN_MODEL, "method": "paired teacher-forced activation differences",
              "system": SYSTEM, "selection_prompts": PROMPTS, "baseline": baseline,
              "dose": args.dose, "motifs": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
