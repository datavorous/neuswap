"""Try a discovered Qwen behavior with any prompt, beside its exact baseline."""

import argparse

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


BEHAVIORS = {
    "specificity": (22, 3357, -1, 0.4),
    "freedom": (22, 5137, -1, 0.4),
    "square": (26, 536, 1, 0.3),
    "opening": (24, 1786, -1, 0.3),
    "comparison": (26, 4923, -1, 0.3),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", choices=tuple(BEHAVIORS), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="Answer directly in one or two clear sentences.")
    parser.add_argument("--dose", type=float,
                        help="Fraction of the prompt's layer residual norm; defaults to the tested dose")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    layer, neuron, sign, default_dose = BEHAVIORS[args.behavior]
    dose = default_dose if args.dose is None else args.dose
    if not 0 < dose <= 1:
        parser.error("Dose must be in (0, 1]")
    messages = [{"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt}]
    calibration = engine.projected_delta_unit(layer, neuron, messages)
    intervention = Intervention(layer, neuron,
                                sign * dose * calibration["delta_per_residual"])
    baseline = engine.generate(messages, max_tokens=args.max_tokens,
                               temperature=args.temperature)["text"]
    steered = engine.generate(messages, [intervention], max_tokens=args.max_tokens,
                              temperature=args.temperature)["text"]
    print("BASELINE:\n" + baseline)
    print(f"\n{args.behavior.upper()} (L{layer} N{neuron} {sign:+}, dose {dose:g}):\n" + steered)


if __name__ == "__main__":
    main()
