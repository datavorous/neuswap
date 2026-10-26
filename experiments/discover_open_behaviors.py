"""Bounded, open-ended Qwen steering screen on varied non-NPC prompts."""

import argparse
import json
import random
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


SOURCE = "https://github.com/lyramakesmusic/neuron-explorer/blob/main/research_log.md"
SEEDS = [
    (22, 2742, 1, "reported alien identity/theme"),
    (22, 2734, 1, "reported interrogatives"),
    (22, 2750, 1, "reported horizontal/vertical language"),
    (22, 2737, 1, "reported ideally/if possible"),
    (22, 5344, -1, "reported Responsible AI shift"),
    (22, 3103, 1, "reported climbing theme"),
    (22, 4002, -1, "reported money/water shift"),
]
PROMPTS = [
    "Who are you?",
    "Explain what a compass does in one sentence.",
    "I found a lost key on the pavement. What should I do?",
]
SYSTEM = "Answer directly in one or two clear sentences."


def messages(user):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/open_behavior_screen.json"))
    parser.add_argument("--relative-dose", type=float, default=0.6)
    parser.add_argument("--random-count", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=468)
    parser.add_argument("--no-reported-seeds", action="store_true")
    parser.add_argument("--random-design", choices=("legacy", "mixed"), default="legacy")
    parser.add_argument("--random-layers", type=int, nargs="+", default=[20, 22, 24, 26])
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    if not 0 < args.relative_dose <= 1 or not 0 <= args.random_count <= 96:
        parser.error("Use dose in (0, 1] and at most 96 random neurons")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    if any(not 0 <= layer < engine.layers for layer in args.random_layers):
        parser.error("Random layer outside model dimensions")
    rng = random.Random(args.random_seed)
    excluded = {(layer, neuron) for layer, neuron, _, _ in SEEDS}
    randoms = []
    for i in range(args.random_count):
        layer = ((8 if i % 2 else 22) if args.random_design == "legacy"
                 else args.random_layers[i % len(args.random_layers)])
        neuron = rng.randrange(engine.width)
        while (layer, neuron) in excluded:
            neuron = rng.randrange(engine.width)
        excluded.add((layer, neuron))
        sign = ((1 if i % 2 else -1) if args.random_design == "legacy"
                else rng.choice((-1, 1)))
        randoms.append((layer, neuron, sign, "random control"))
    candidates = ([] if args.no_reported_seeds else SEEDS) + randoms
    baselines = {prompt: engine.generate(messages(prompt), max_tokens=args.max_tokens,
                                         temperature=0)["text"] for prompt in PROMPTS}
    rows = []
    for layer, neuron, sign, provenance in candidates:
        outputs = {}
        calibrations = {}
        for prompt in PROMPTS:
            calibration = engine.projected_delta_unit(layer, neuron, messages(prompt))
            delta = sign * args.relative_dose * calibration["delta_per_residual"]
            reply = engine.generate(messages(prompt), [Intervention(layer, neuron, delta)],
                                    max_tokens=args.max_tokens, temperature=0)
            outputs[prompt] = reply["text"]
            calibrations[prompt] = {"delta": delta, **calibration}
        row = {"layer": layer, "neuron": neuron, "sign": sign,
               "provenance": provenance, "outputs": outputs,
               "calibrations": calibrations,
               "changed_prompts": sum(outputs[p] != baselines[p] for p in PROMPTS)}
        rows.append(row)
        print(f"L{layer} N{neuron} {sign:+}: {row['changed_prompts']}/{len(PROMPTS)} changed", flush=True)
    result = {"model": QWEN_MODEL, "source": SOURCE, "phase": "screen",
              "relative_dose": args.relative_dose, "max_tokens": args.max_tokens,
              "random_seed": args.random_seed,
              "random_design": args.random_design,
              "random_layers": args.random_layers,
              "system": SYSTEM, "prompts": PROMPTS, "baselines": baselines,
              "candidates": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
