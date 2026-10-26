"""Bounded sparse MLP-set and MLP-output-direction search on Qwen 3.5 4B.

Selection is separate from validation; grade selection outputs before validating.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from data import SELECTION, TRAIN, VALIDATION
from model import AttentionShift, Intervention, OutputShift, TheatreModel
from search_evil_funny import CONTRASTS, PROMPT_ONLY, QWEN_MODEL, TRAITS, messages


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def config_interventions(config):
    if config["kind"] == "sparse_mlp":
        return [Intervention(**row) for row in config["interventions"]], [], []
    vector = np.asarray(config["vector"], dtype=np.float32)
    if config["kind"] == "attention_output_direction":
        return [], [], [AttentionShift(config["layer"], vector)]
    return [], [OutputShift(config["layer"], vector)], []


def candidates(hidden_diffs, hidden_values, output_diffs, attention_diffs):
    """Select two layers per method by consistency across contrast pairs."""
    diff = np.stack(hidden_diffs)
    mean = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1)
    consistency = (np.sign(diff) == np.sign(mean)).mean(axis=0)
    rank_score = np.abs(mean) / np.sqrt(sd + 0.03) * consistency
    native_sd = np.stack(hidden_values).std(axis=0, ddof=1).clip(0.1)
    layer_scores = []
    for layer in range(mean.shape[0]):
        top = np.argsort(rank_score[layer])[-32:]
        layer_scores.append(float(rank_score[layer, top].mean()))
    sparse_layers = np.argsort(layer_scores)[-2:][::-1]
    configs = []
    for layer in sparse_layers:
        order = np.argsort(rank_score[layer])[::-1]
        for count in (8, 32):
            coords = order[:count]
            for dose in (1.0, 2.0):
                rows = [{"layer": int(layer), "neuron": int(n),
                         "delta": float(np.sign(mean[layer, n]) * native_sd[layer, n] * dose)}
                        for n in coords]
                configs.append({"id": f"sparse_L{layer}_K{count}_D{dose:g}",
                                "kind": "sparse_mlp", "layer": int(layer),
                                "count": count, "dose": dose, "interventions": rows})

    direction_rankings = {}
    for kind, diffs, prefix in (("mlp_output_direction", output_diffs, "output"),
                                ("attention_output_direction", attention_diffs, "attention")):
        out = np.stack(diffs)
        out_mean = out.mean(axis=0)
        norms = np.linalg.norm(out, axis=-1)
        directional_consistency = np.linalg.norm(out_mean, axis=-1) / (norms.mean(axis=0) + 1e-6)
        direction_rankings[kind] = directional_consistency.tolist()
        for layer in np.argsort(directional_consistency)[-2:][::-1]:
            for dose in (1.0, 2.0, 4.0):
                configs.append({"id": f"{prefix}_L{layer}_D{dose:g}",
                                "kind": kind, "layer": int(layer),
                                "dose": dose, "vector": (out_mean[layer] * dose).tolist(),
                                "direction_norm": float(np.linalg.norm(out_mean[layer]))})
    ranking = {"sparse_layer_scores": layer_scores,
               "directional_consistency": direction_rankings}
    return configs, ranking


def random_controls(selected, width, residual_width, count=4):
    rng = random.Random(37)
    controls = []
    if selected["kind"] == "sparse_mlp":
        magnitudes = [row["delta"] for row in selected["interventions"]]
        excluded = {row["neuron"] for row in selected["interventions"]}
        for i in range(count):
            coords = rng.sample([n for n in range(width) if n not in excluded], len(magnitudes))
            controls.append({"id": f"random_{i}", "kind": "sparse_mlp",
                             "layer": selected["layer"],
                             "interventions": [{"layer": selected["layer"], "neuron": n,
                                                "delta": delta}
                                               for n, delta in zip(coords, magnitudes)]})
    else:
        norm = np.linalg.norm(selected["vector"])
        for i in range(count):
            vector = np.array([rng.gauss(0, 1) for _ in range(residual_width)], dtype=np.float32)
            vector *= norm / np.linalg.norm(vector)
            controls.append({"id": f"random_{i}", "kind": selected["kind"],
                             "layer": selected["layer"], "vector": vector.tolist()})
    return controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", choices=tuple(TRAITS), required=True)
    parser.add_argument("--phase", choices=("search", "validate"), default="search")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    if args.phase == "validate":
        result = json.loads(args.output.read_text())
        if result["phase"] != "selection_complete":
            parser.error("Run search before validation")
        engine = TheatreModel.from_cache(result["model"])
        selected = result["traits"][args.trait]["selected"]
        controls = random_controls(selected, engine.width, engine.residual_width)
        rows = []
        for scenario in VALIDATION:
            for config in [selected, *controls]:
                interventions, shifts, attention_shifts = config_interventions(config)
                output = engine.generate(messages(scenario), interventions,
                                         output_shifts=shifts, attention_shifts=attention_shifts,
                                         max_tokens=result["max_tokens"],
                                         temperature=result["temperature"])
                rows.append({"scenario": scenario.id,
                             "condition": "steered" if config is selected else config["id"],
                             "text": output["text"],
                             "generated_tokens": output["generated_tokens"]})
            output = engine.generate(messages(scenario, PROMPT_ONLY[args.trait]),
                                     max_tokens=result["max_tokens"],
                                     temperature=result["temperature"])
            rows.append({"scenario": scenario.id, "condition": "prompt_only", "text": output["text"]})
        result["traits"][args.trait]["validation_rows"] = rows
        result["traits"][args.trait]["random_controls"] = controls
        result["baseline"] = [{"scenario": s.id,
                               "text": engine.generate(messages(s), max_tokens=result["max_tokens"],
                                                       temperature=result["temperature"])["text"]}
                              for s in VALIDATION]
        result["phase"] = "validation_complete"
        save(args.output, result)
        print("saved", args.output, flush=True)
        return

    engine = TheatreModel.from_cache(args.model)
    hidden_diffs, hidden_values, output_diffs, attention_diffs = [], [], [], []
    for scenario in TRAIN:
        neutral, target = CONTRASTS[scenario.id][0], CONTRASTS[scenario.id][TRAITS[args.trait]]
        limit = min(len(engine.tokenizer.encode(neutral, add_special_tokens=False)),
                    len(engine.tokenizer.encode(target, add_special_tokens=False)), 24)
        neutral_hidden, neutral_out, neutral_attn = engine.activation_means_with_components(
            messages(scenario), neutral, limit)
        target_hidden, target_out, target_attn = engine.activation_means_with_components(
            messages(scenario), target, limit)
        hidden_diffs.append(target_hidden - neutral_hidden)
        hidden_values.extend((neutral_hidden, target_hidden))
        output_diffs.append(target_out - neutral_out)
        attention_diffs.append(target_attn - neutral_attn)
        print("captured", scenario.id, flush=True)
    configs, ranking = candidates(hidden_diffs, hidden_values, output_diffs, attention_diffs)
    baseline = {s.id: engine.generate(messages(s), max_tokens=args.max_tokens,
                                      temperature=args.temperature)["text"] for s in SELECTION}
    for config in configs:
        interventions, shifts, attention_shifts = config_interventions(config)
        config["selection_outputs"] = {s.id: engine.generate(
            messages(s), interventions, output_shifts=shifts, attention_shifts=attention_shifts,
            max_tokens=args.max_tokens, temperature=args.temperature)["text"] for s in SELECTION}
        print(config["id"], flush=True)
    result = {"model": args.model, "temperature": args.temperature,
              "max_tokens": args.max_tokens, "phase": "selection_complete",
              "train": [s.id for s in TRAIN], "selection": [s.id for s in SELECTION],
              "validation": [s.id for s in VALIDATION],
              "traits": {args.trait: {"ranking": ranking, "selection_baseline": baseline,
                                       "selection_trials": configs, "selected": configs[0],
                                       "selection_rule": "await blind judge grading"}}}
    save(args.output, result)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
