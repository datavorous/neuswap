"""Bounded contrastive search for exploratory NPC neuron controls."""

import argparse
import json
from pathlib import Path

import numpy as np

from data import SELECTION, TRAIN, TRAITS
from metrics import assess, trait_score
from model import DEFAULT_MODEL, Intervention, TheatreModel


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def rank(differences, activations, count):
    diff = np.stack(differences)
    mean = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1)
    consistency = (np.sign(diff) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) / np.sqrt(sd + 0.03) * consistency
    native_std = np.stack(activations).std(axis=0, ddof=1)
    order = np.argsort(score.ravel())[::-1]
    width = score.shape[1]
    selected = []
    per_layer = {}
    for flat in order:
        layer, neuron = divmod(int(flat), width)
        if per_layer.get(layer, 0) >= 2 or abs(mean[layer, neuron]) < 0.08:
            continue
        selected.append({"layer": layer, "neuron": neuron,
                         "rank_score": float(score[layer, neuron]),
                         "mean_difference": float(mean[layer, neuron]),
                         "native_std": float(max(native_std[layer, neuron], 0.1)),
                         "sign_consistency": float(consistency[layer, neuron])})
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(selected) == count:
            break
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("results/discovery.json"))
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(args.model)
    result = {"model": args.model, "train_scenarios": [s.id for s in TRAIN],
              "selection_scenarios": [s.id for s in SELECTION],
              "candidate_budget_per_trait": args.candidates, "traits": {},
              "method": "Post-SwiGLU mean activations on matched answer-token spans; native per-neuron standard deviation sets steering units."}
    neutral = {}
    positives = {trait: [] for trait in TRAITS}
    all_activations = {trait: [] for trait in TRAITS}
    pairs = []
    for scenario in TRAIN:
        baseline = engine.generate(scenario.messages(), max_tokens=args.max_tokens)["text"]
        neutral[scenario.id] = baseline
        pairs.append({"scenario": scenario.id, "neutral": baseline,
                      "positive": scenario.positive})
        for trait in TRAITS:
            target = scenario.positive[trait]
            neutral_tokens = engine.tokenizer.encode(baseline, add_special_tokens=False)
            target_tokens = engine.tokenizer.encode(target, add_special_tokens=False)
            limit = min(len(neutral_tokens), len(target_tokens), 24)
            base_activation = engine.activation_means(scenario.messages(), baseline, limit)
            trait_activation = engine.activation_means(scenario.messages(), target, limit)
            positives[trait].append(trait_activation - base_activation)
            all_activations[trait].extend((base_activation, trait_activation))
        print(f"captured {scenario.id}", flush=True)
    result["contrastive_pairs"] = pairs
    selection_baseline = {s.id: engine.generate(s.messages(), max_tokens=args.max_tokens)["text"]
                          for s in SELECTION}
    result["selection_baseline"] = selection_baseline
    for trait in TRAITS:
        candidates = rank(positives[trait], all_activations[trait], args.candidates)
        trials = []
        for candidate in candidates:
            candidate["projected_calibration"] = engine.projected_delta_unit(
                candidate["layer"], candidate["neuron"], SELECTION[0].messages())
            doses = [("native_std", multiple, multiple * candidate["native_std"])
                     for multiple in (-4.0, -2.0, 2.0, 4.0)]
            doses += [("residual_fraction", fraction,
                       fraction * candidate["projected_calibration"]["delta_per_residual"])
                      for fraction in (-0.1, -0.05, 0.05, 0.1)]
            for calibration, dose, delta in doses:
                intervention = Intervention(candidate["layer"], candidate["neuron"], delta)
                for scenario in SELECTION:
                    output = engine.generate(scenario.messages(), [intervention],
                                             max_tokens=args.max_tokens)
                    assessment = assess(scenario, output["text"])
                    trials.append({"coordinate": [candidate["layer"], candidate["neuron"]],
                                   "calibration": calibration, "dose": dose, "delta": delta,
                                   "scenario": scenario.id, "text": output["text"],
                                   "assessment": assessment,
                                   "trait_score": trait_score(trait, assessment)})
        conditions = {}
        for row in trials:
            key = (*row["coordinate"], row["calibration"], row["dose"])
            conditions.setdefault(key, []).append(row)
        best_key = max(conditions, key=lambda k: (sum(r["trait_score"] for r in conditions[k]),
                                                   sum(r["assessment"]["relevance_proxy"] for r in conditions[k]),
                                                   sum(r["assessment"]["fluency_proxy"] for r in conditions[k])))
        chosen = next(c for c in candidates if (c["layer"], c["neuron"]) == best_key[:2])
        selected_trial = conditions[best_key][0]
        result["traits"][trait] = {"candidates": candidates,
                                    "selected": {**chosen, "calibration": best_key[2],
                                                 "dose": best_key[3],
                                                 "delta": selected_trial["delta"],
                                                 "status": "exploratory"},
                                    "selection_scores": {str(k): sum(r["trait_score"] for r in v)
                                                         for k, v in conditions.items()},
                                    "selection_trials": trials}
        save(args.output, result)
        print(f"selected exploratory {trait}: L{best_key[0]} N{best_key[1]} "
              f"{best_key[2]} {best_key[3]:+g}; proxy {sum(r['trait_score'] for r in conditions[best_key])}/3",
              flush=True)
    save(args.output, result)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
