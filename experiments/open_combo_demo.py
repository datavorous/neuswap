"""Compare baseline, each component, and the frozen two-neuron pair on one prompt."""

import argparse
import json

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


SYSTEM = "Answer naturally in one or two coherent sentences."
COMPONENTS = ((8, 8437), (8, 8051))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=SYSTEM)
    parser.add_argument("--dose", type=float, default=0.33)
    parser.add_argument("--max-tokens", type=int, default=68)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    msg = [{"role": "system", "content": args.system},
           {"role": "user", "content": args.prompt}]
    baseline = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    active = []
    details = []
    for layer, neuron in COMPONENTS:
        cal = engine.projected_delta_unit(layer, neuron, msg)
        delta = args.dose * cal["delta_per_residual"]
        active.append(Intervention(layer, neuron, delta))
        details.append({"layer": layer, "neuron": neuron, "delta": delta})
    first = engine.generate(msg, interventions=active[:1], temperature=0,
                            max_tokens=args.max_tokens)["text"]
    second = engine.generate(msg, interventions=active[1:], temperature=0,
                             max_tokens=args.max_tokens)["text"]
    pair = engine.generate(msg, interventions=active, temperature=0,
                           max_tokens=args.max_tokens)["text"]
    restored = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    assert restored == baseline
    print(json.dumps({"prompt": args.prompt, "dose_each": args.dose,
                      "components": details, "baseline": baseline,
                      "first_only": first, "second_only": second,
                      "pair": pair, "baseline_restored_exact": True},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
