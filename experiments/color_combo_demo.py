"""Show baseline, two teal-neuron singles, and their pair on the same prompt."""

import argparse
import json
from pathlib import Path

from color_choice_gradient import messages
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


CONFIG = Path("results/color_sparse_combo/configurations.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    msg = messages(args.prompt)
    pieces = config["components"]
    active = []
    for item in pieces:
        calibration = engine.projected_delta_unit(item["layer"], item["neuron"], msg)
        delta = item["signed_dose"] * calibration["delta_per_residual"]
        active.append(Intervention(item["layer"], item["neuron"], delta))
    outputs = {}
    for name, interventions in (("baseline", []), ("L30 alone", active[:1]),
                                ("L31 alone", active[1:]), ("pair", active),
                                ("restored", [])):
        outputs[name] = engine.generate(msg, interventions=interventions,
                                        temperature=0, max_tokens=args.max_tokens)["text"]
    if outputs["baseline"] != outputs["restored"]:
        raise RuntimeError("Baseline failed to restore exactly")
    for name in ("baseline", "L30 alone", "L31 alone", "pair"):
        print(f"{name.upper()}\n{outputs[name]}\n")


if __name__ == "__main__":
    main()
