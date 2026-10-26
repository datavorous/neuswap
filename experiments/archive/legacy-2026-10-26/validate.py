"""Held-out randomized comparison for the best pilot intervention."""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from neurofuzz import attach_probes, chat_prompt, model_path, tic_metrics


PROMPTS = [
    "Write a polished 180-word essay on why local journalism matters. Make it engaging.",
    "Write a reflective 180-word paragraph about learning a musical instrument.",
    "Write a vivid, polished 180-word op-ed arguing for safer sidewalks.",
    "Write a polished 180-word explanation of why clean drinking water matters.",
    "Write an engaging 180-word essay on the value of scientific museums.",
    "Write a reflective 180-word paragraph on caring for an aging parent.",
    "Write a persuasive 180-word paragraph about protecting coastal habitats.",
    "Write a polished 180-word explanation of how teamwork improves a project.",
]


def rate(rows):
    metrics = [tic_metrics(r["answer"]) for r in rows]
    words = sum(m["words"] for m in metrics)
    return 1000 * sum(m["tic_score"] for m in metrics) / max(words, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/pilot.json"))
    parser.add_argument("--output", type=Path, default=Path("results/validation.json"))
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=220)
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_text())
    # Selection uses the pilot evaluation prompts only; these validation
    # prompts have never been used to rank or select the neuron.
    best = min(pilot["trials"], key=lambda t: rate(t["rows"]))
    candidate = best["candidate"]
    layer, neuron = candidate["layer"], candidate["neuron"]
    delta = best["multiplier"] * candidate["activation_difference"]
    model, tokenizer = load(model_path(pilot["model"]))
    _, interventions = attach_probes(model)
    width = int(model.model.layers[layer].mlp.base.gate_proj.weight.shape[0])
    rng = random.Random(42)
    control_neuron = rng.choice([i for i in range(width) if i != neuron])
    variants = {
        "baseline": ({}, ""),
        "steered": ({layer: {neuron: delta}}, ""),
        "random_neuron": ({layer: {control_neuron: delta}}, ""),
        "prompt_only": ({}, " Avoid em dashes, 'not X but Y' contrasts, and repeated sentence openings."),
    }
    rows = []
    sampler = make_sampler(temp=0.7, top_p=0.9)
    for seed in range(args.seeds):
        for question in PROMPTS:
            for name, (steer, instruction) in variants.items():
                interventions.clear()
                interventions.update(steer)
                mx.random.seed(seed)
                answer = generate(model, tokenizer, chat_prompt(tokenizer, question + instruction),
                                  max_tokens=args.max_tokens, sampler=sampler, verbose=False)
                rows.append({"seed": seed, "prompt": question, "variant": name,
                             "answer": answer, "metrics": tic_metrics(answer)})
            print(f"validated seed {seed}: {question[:38]}", flush=True)
    interventions.clear()
    summary = {name: {"tic_per_1000_words": rate([r for r in rows if r["variant"] == name]),
                      "words": sum(r["metrics"]["words"] for r in rows if r["variant"] == name),
                      "tic_count": sum(r["metrics"]["tic_score"] for r in rows if r["variant"] == name)}
               for name in variants}
    output = {"selected": {"layer": layer, "neuron": neuron, "delta": delta},
              "random_control": {"layer": layer, "neuron": control_neuron, "delta": delta},
              "seeds": args.seeds, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
