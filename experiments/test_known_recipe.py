"""Fixed Qwen3.5-4B two-neuron recipe, with held-out random and prompt controls."""

import argparse
import json
import random
from pathlib import Path

from data import TRAIN, VALIDATION
from model import Intervention, TheatreModel
from search_evil_funny import PROMPT_ONLY, QWEN_MODEL, messages


RECIPE = ((14, 8949, 1.0), (13, 7956, -1.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/qwen_known_recipe_dynamic.json"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--static-calibration", action="store_true",
                        help="Use one training prompt's scale for all scenarios")
    args = parser.parse_args()
    if not 0 < args.dose <= 2:
        parser.error("Dose must be in (0, 2]")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    static_calibration = ([engine.projected_delta_unit(layer, neuron, messages(TRAIN[0]))
                           for layer, neuron, _ in RECIPE] if args.static_calibration else None)
    rng = random.Random(2026)
    random_coords = [[rng.choice([n for n in range(engine.width) if n != neuron])
                      for layer, neuron, _ in RECIPE] for _ in range(4)]
    rows = []
    baseline = []
    calibrations = {}
    interventions_by_scenario = {}
    for scenario in VALIDATION:
        context = messages(scenario)
        calibration = static_calibration or [engine.projected_delta_unit(layer, neuron, context)
                                             for layer, neuron, _ in RECIPE]
        calibrations[scenario.id] = calibration
        selected = [Intervention(layer, neuron, sign * item["delta_per_residual"] * args.dose)
                    for (layer, neuron, sign), item in zip(RECIPE, calibration)]
        interventions_by_scenario[scenario.id] = [vars(item) for item in selected]
        randoms = [[Intervention(layer, random_coords[i][j], selected[j].delta)
                    for j, (layer, neuron, _) in enumerate(RECIPE)] for i in range(4)]
        baseline.append({"scenario": scenario.id,
                         "text": engine.generate(context, max_tokens=args.max_tokens,
                                                 temperature=args.temperature)["text"]})
        for condition, offsets in [("steered", selected),
                                   *((f"random_{i}", item) for i, item in enumerate(randoms))]:
            output = engine.generate(context, offsets, max_tokens=args.max_tokens,
                                     temperature=args.temperature)
            rows.append({"scenario": scenario.id, "condition": condition,
                         "text": output["text"], "generated_tokens": output["generated_tokens"],
                         "seconds": output["seconds"]})
        prompt = engine.generate(messages(scenario, PROMPT_ONLY["villainy"]),
                                 max_tokens=args.max_tokens, temperature=args.temperature)
        rows.append({"scenario": scenario.id, "condition": "prompt_only",
                     "text": prompt["text"], "generated_tokens": prompt["generated_tokens"]})
        print("generated", scenario.id, flush=True)
    result = {"model": QWEN_MODEL, "phase": "validation_complete", "method": "published_two_neuron_recipe",
              "source": "https://x.com/_lyraaaa_/status/2113833330429083662",
              "recipe": RECIPE, "calibration_mode": "static" if args.static_calibration else "per_input",
              "dose": args.dose,
              "calibration": calibrations, "interventions": interventions_by_scenario,
              "random_control_coordinates": random_coords,
              "max_tokens": args.max_tokens, "temperature": args.temperature,
              "baseline": baseline,
              "traits": {"villainy": {"selected": {"id": "published_L14_N8949_L13_N7956"},
                                      "validation_rows": rows}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
