"""Held-out comparison of the best sparse, MLP-output, and attention-output probes."""

import argparse
import json
from pathlib import Path

from data import VALIDATION
from model import TheatreModel
from search_directions import config_interventions, random_controls
from search_evil_funny import PROMPT_ONLY, messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trait", choices=("villainy", "comedy"), required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    trials = source["traits"][args.trait]["selection_trials"]
    engine = TheatreModel.from_cache(source["model"])
    selected = {}
    controls = {}
    for kind in ("sparse_mlp", "mlp_output_direction", "attention_output_direction"):
        group = [x for x in trials if x["kind"] == kind]
        selected[kind] = max(group, key=lambda x: (x["judge_persona_hits"],
                                                    x["judge_successes"], x["judge_target_sum"]))
        controls[kind] = random_controls(selected[kind], engine.width,
                                         engine.residual_width, count=2)
    rows = []
    baseline = []
    for scenario in VALIDATION:
        baseline.append({"scenario": scenario.id, "text": engine.generate(
            messages(scenario), max_tokens=source["max_tokens"],
            temperature=source["temperature"])["text"]})
        for kind, config in selected.items():
            for condition, item in [(kind, config),
                                    *((f"{kind}_random_{i}", c)
                                      for i, c in enumerate(controls[kind]))]:
                interventions, shifts, attention = config_interventions(item)
                output = engine.generate(messages(scenario), interventions,
                                         output_shifts=shifts, attention_shifts=attention,
                                         max_tokens=source["max_tokens"],
                                         temperature=source["temperature"])
                rows.append({"scenario": scenario.id, "condition": condition,
                             "text": output["text"]})
        prompt = engine.generate(messages(scenario, PROMPT_ONLY[args.trait]),
                                 max_tokens=source["max_tokens"],
                                 temperature=source["temperature"])
        rows.append({"scenario": scenario.id, "condition": "prompt_only", "text": prompt["text"]})
        print("generated", scenario.id, flush=True)
    result = {"model": source["model"], "phase": "validation_complete",
              "method": "top_per_intervention_family", "source": str(args.input),
              "selected": selected, "controls": controls, "baseline": baseline,
              "traits": {args.trait: {"validation_rows": rows}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
