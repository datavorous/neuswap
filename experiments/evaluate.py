"""Held-out comparisons for exploratory Neural Theatre controls."""

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

from data import TRAITS, VALIDATION
from metrics import assess, trait_score
from model import Intervention, TheatreModel


def summarize(rows, trait):
    return {"n": len(rows), "trait_proxy_rate": sum(trait_score(trait, r["assessment"]) for r in rows) / len(rows),
            "relevance_proxy_rate": sum(r["assessment"]["relevance_proxy"] for r in rows) / len(rows),
            "fluency_proxy_rate": sum(r["assessment"]["fluency_proxy"] for r in rows) / len(rows),
            "mean_words": statistics.mean(r["assessment"]["words"] for r in rows),
            "median_seconds": statistics.median(r["seconds"] for r in rows),
            "median_generation_tps": statistics.median(r["generation_tps"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=Path("results/discovery.json"))
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    parser.add_argument("--random-controls", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    discovery = json.loads(args.discovery.read_text())
    engine = TheatreModel.from_cache(discovery["model"])
    started = perf_counter()
    rows = []
    baseline = {}
    for scenario in VALIDATION:
        output = engine.generate(scenario.messages(), max_tokens=args.max_tokens)
        baseline[scenario.id] = output
        rows.append({"trait": None, "condition": "baseline", "scenario": scenario.id,
                     **output, "assessment": assess(scenario, output["text"])})
    result = {"model": discovery["model"], "validation_scenarios": [s.id for s in VALIDATION],
              "random_controls_per_trait": args.random_controls, "rows": rows,
              "traits": {}, "status": "exploratory; proxy scores require human review"}
    rng = random.Random(17)
    switch_times = []
    model_id = id(engine.model)
    for trait, instruction in TRAITS.items():
        selected = discovery["traits"][trait]["selected"]
        layer, neuron, delta = selected["layer"], selected["neuron"], selected["delta"]
        controls = rng.sample([n for n in range(engine.width) if n != neuron], args.random_controls)
        conditions = {"steered": [Intervention(layer, neuron, delta)]}
        for n in controls:
            conditions[f"random_N{n}"] = [Intervention(layer, n, delta)]
        for scenario in VALIDATION:
            output = engine.generate(scenario.messages(instruction), max_tokens=args.max_tokens)
            rows.append({"trait": trait, "condition": "prompt_only", "scenario": scenario.id,
                         **output, "assessment": assess(scenario, output["text"])})
            for condition, interventions in conditions.items():
                output = engine.generate(scenario.messages(), interventions,
                                         max_tokens=args.max_tokens)
                rows.append({"trait": trait, "condition": condition, "scenario": scenario.id,
                             **output, "assessment": assess(scenario, output["text"])})
                t0 = perf_counter()
                with engine.using(interventions):
                    pass
                switch_times.append(perf_counter() - t0)
        baseline_rows = [r for r in rows if r["condition"] == "baseline"]
        trait_rows = [r for r in rows if r["trait"] == trait]
        summaries = {name: summarize([r for r in trait_rows if r["condition"] == name], trait)
                     for name in ["prompt_only", *conditions]}
        summaries["baseline"] = summarize(baseline_rows, trait)
        random_scores = [summaries[name]["trait_proxy_rate"] for name in conditions if name.startswith("random_")]
        result["traits"][trait] = {"selected": selected, "random_neurons": controls,
                                    "summary": summaries,
                                    "selected_exceeds_all_random_proxies": summaries["steered"]["trait_proxy_rate"] > max(random_scores),
                                    "status": "exploratory; no automatic validation"}
        result["rows"] = rows
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"evaluated {trait}: baseline {summaries['baseline']['trait_proxy_rate']:.2f}, "
              f"steered {summaries['steered']['trait_proxy_rate']:.2f}, "
              f"random max {max(random_scores):.2f}, "
              f"prompt {summaries['prompt_only']['trait_proxy_rate']:.2f}", flush=True)
    result["elapsed_seconds"] = perf_counter() - started
    result["model_reloaded"] = id(engine.model) != model_id
    result["median_switch_overhead_seconds"] = statistics.median(switch_times)
    result["peak_gb"] = max(r["peak_gb"] for r in rows)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
