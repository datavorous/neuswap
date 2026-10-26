"""Selection-only ablation and dose screen for the reported Qwen evil pair."""

import argparse
import json
from pathlib import Path

from data import SELECTION
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL, messages
from test_known_recipe import RECIPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/qwen_known_pair_ablation.json"))
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    calibrations = {s.id: [engine.projected_delta_unit(layer, neuron, messages(s))
                           for layer, neuron, _ in RECIPE] for s in SELECTION}
    baseline = {s.id: engine.generate(messages(s), max_tokens=args.max_tokens,
                                      temperature=0.7)["text"] for s in SELECTION}
    trials = []
    for active in ((0,), (1,), (0, 1)):
        for dose in (0.25, 0.5, 1.0):
            config = {"id": f"pair_{''.join(str(i) for i in active)}_D{dose:g}",
                      "kind": "known_pair_ablation", "active_neurons": list(active),
                      "dose": dose, "selection_outputs": {}}
            for scenario in SELECTION:
                offsets = [Intervention(layer, neuron,
                                        sign * calibrations[scenario.id][i]["delta_per_residual"] * dose)
                           for i, (layer, neuron, sign) in enumerate(RECIPE) if i in active]
                config["selection_outputs"][scenario.id] = engine.generate(
                    messages(scenario), offsets, max_tokens=args.max_tokens,
                    temperature=0.7)["text"]
            trials.append(config)
            print(config["id"], flush=True)
    result = {"model": QWEN_MODEL, "method": "known_pair_ablation",
              "phase": "selection_complete", "recipe": RECIPE,
              "calibration": calibrations, "max_tokens": args.max_tokens, "temperature": 0.7,
              "traits": {"villainy": {"selection_baseline": baseline,
                                      "selection_trials": trials,
                                      "selected": trials[0],
                                      "selection_rule": "await blind judge grading"}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
