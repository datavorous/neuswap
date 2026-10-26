"""Validate open-ended Qwen neuron effects on unseen general-purpose prompts."""

import argparse
import json
import random
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


SYSTEM = "Answer directly in one or two clear sentences."
PROMPTS = [
    "How would you spend a free afternoon?",
    "Explain what a garden needs to thrive.",
    "What is the first step in preparing for a long trip?",
    "Describe yourself in one sentence.",
    "Write a short greeting to a new neighbor.",
    "How should I organize a cluttered desk?",
]
NARRATIVE_SYSTEM = "You are a fictional advisor with a distinctive voice. Reply in one or two sentences."
NARRATIVE_PROMPTS = [
    "A small town plans to replace its old bridge. What would you tell them?",
    "Someone asks how to begin learning the violin. What do you say?",
    "A friend wants to leave a boring job. What advice do you give?",
    "A traveler sees a door painted blue. What might be behind it?",
    "Write a one-sentence toast at a dinner.",
]
FRESH_PROMPTS = [
    "How should I prepare for a first pottery class?",
    "Explain why a bicycle stays balanced while moving.",
    "What could I say to someone who lost a board game?",
    "Write one sentence introducing a new science museum exhibit.",
    "I have an hour before my train. How should I use it?",
    "A community library asks for a short welcome message. Write one.",
]
CREATIVE_SYSTEM = "Write vividly, but keep the reply coherent and directly responsive in one or two sentences."
CREATIVE_PROMPTS = [
    "Describe a mysterious café in two sentences.",
    "Invent a friendly mascot for a neighborhood market.",
    "Suggest a theme for a children's science fair.",
    "Write a short invitation to a rooftop picnic.",
    "Describe a strange souvenir found in a suitcase.",
    "Write a one-sentence ending to a gentle adventure.",
]


def messages(prompt, system=SYSTEM):
    return [{"role": "system", "content": system}, {"role": "user", "content": prompt}]


def parse_candidate(value):
    try:
        layer, neuron, sign = map(int, value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("Use LAYER:NEURON:SIGN, e.g. 22:3357:-1") from None
    if sign not in (-1, 1):
        raise argparse.ArgumentTypeError("SIGN must be -1 or 1")
    return layer, neuron, sign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doses", type=float, nargs="+", default=[0.4, 0.6])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--random-controls", type=int, default=2)
    parser.add_argument("--prompt-pack", choices=("general", "narrative", "fresh", "creative"), default="general")
    args = parser.parse_args()
    if any(not 0 < dose <= 1 for dose in args.doses):
        parser.error("Doses must be in (0, 1]")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    prompts = {"general": PROMPTS, "narrative": NARRATIVE_PROMPTS,
               "fresh": FRESH_PROMPTS, "creative": CREATIVE_PROMPTS}[args.prompt_pack]
    system = {"general": SYSTEM, "narrative": NARRATIVE_SYSTEM,
              "fresh": SYSTEM, "creative": CREATIVE_SYSTEM}[args.prompt_pack]
    for layer, neuron, _ in args.candidate:
        if not 0 <= layer < engine.layers or not 0 <= neuron < engine.width:
            parser.error("Candidate outside model dimensions")
    baseline = {prompt: engine.generate(messages(prompt, system), max_tokens=args.max_tokens,
                                        temperature=0)["text"] for prompt in prompts}
    rng = random.Random(734)
    rows = []
    excluded = {(layer, neuron) for layer, neuron, _ in args.candidate}
    control_specs = []
    for layer, neuron, sign in args.candidate:
        for _ in range(args.random_controls):
            n = rng.randrange(engine.width)
            while (layer, n) in excluded:
                n = rng.randrange(engine.width)
            excluded.add((layer, n))
            control_specs.append((layer, n, sign, (layer, neuron, sign)))
    specs = [(layer, neuron, sign, None) for layer, neuron, sign in args.candidate] + control_specs
    for layer, neuron, sign, control_for in specs:
        for dose in ([args.doses[0]] if control_for else args.doses):
            outputs, calibrations = {}, {}
            for prompt in prompts:
                calibration = engine.projected_delta_unit(layer, neuron, messages(prompt, system))
                delta = sign * dose * calibration["delta_per_residual"]
                outputs[prompt] = engine.generate(
                    messages(prompt, system), [Intervention(layer, neuron, delta)],
                    max_tokens=args.max_tokens, temperature=0)["text"]
                calibrations[prompt] = {"delta": delta, **calibration}
            rows.append({"layer": layer, "neuron": neuron, "sign": sign,
                         "dose": dose, "control_for": control_for,
                         "outputs": outputs, "calibrations": calibrations})
            print(f"L{layer} N{neuron} {sign:+} dose {dose:g} done", flush=True)
    result = {"model": QWEN_MODEL, "phase": "validation", "system": system,
              "prompt_pack": args.prompt_pack, "prompts": prompts, "max_tokens": args.max_tokens,
              "baseline": baseline, "candidates": args.candidate,
              "random_controls": args.random_controls, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
