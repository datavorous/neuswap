"""Compare one Qwen color-choice steering preset with its exact baseline."""

import argparse
import json
from pathlib import Path

from color_choice_gradient import messages
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


CONFIG = Path("results/color_gradient/configurations.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", choices=("blue", "black", "teal"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--dose", type=float, default=None,
                        help="Prompt-relative dose; defaults to the tested 0.08")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    setting = next(x for x in config["settings"] if x["name"].startswith(args.color + "_"))
    dose = setting["dose"] if args.dose is None else args.dose
    if not 0 <= dose <= .2:
        parser.error("Dose must be between 0 and 0.2")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    msg = messages(args.prompt)
    baseline = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    calibration = engine.projected_delta_unit(setting["layer"], setting["neuron"], msg)
    delta = setting["sign"] * dose * calibration["delta_per_residual"]
    steered = engine.generate(msg, interventions=[Intervention(setting["layer"], setting["neuron"], delta)],
                              temperature=0, max_tokens=args.max_tokens)["text"]
    restored = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    if baseline != restored:
        raise RuntimeError("Baseline did not restore exactly after steering")
    print("BASELINE\n" + baseline + "\n")
    print(f"{args.color.upper()} L{setting['layer']} N{setting['neuron']} dose {dose:g}, activation delta {delta:.3f}\n" + steered)


if __name__ == "__main__":
    main()
