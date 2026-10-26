"""Small 2x2 causal test for interaction between two candidate neurons."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from neurofuzz import attach_probes, chat_prompt, model_path, tic_metrics
from validate import rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/pilot.json"))
    parser.add_argument("--validation", type=Path, default=Path("results/validation.json"))
    parser.add_argument("--output", type=Path, default=Path("results/interactions.json"))
    parser.add_argument("--max-tokens", type=int, default=220)
    args = parser.parse_args()
    pilot = json.loads(args.pilot.read_text())
    validation = json.loads(args.validation.read_text())
    first = validation["selected"]
    second_trial = next(t for t in sorted(pilot["trials"], key=lambda t: rate(t["rows"]))
                        if (t["candidate"]["layer"], t["candidate"]["neuron"])
                        != (first["layer"], first["neuron"]))
    c = second_trial["candidate"]
    second = {"layer": c["layer"], "neuron": c["neuron"],
              "delta": c["activation_difference"] * second_trial["multiplier"]}
    model, tokenizer = load(model_path(pilot["model"]))
    _, interventions = attach_probes(model)
    sampler = make_sampler(temp=0.7, top_p=0.9)
    anchor = [r for r in validation["rows"] if r["variant"] in ("baseline", "steered")]
    rows = []
    for row in anchor:
        if row["variant"] != "baseline":
            continue
        pair_change = {first["layer"]: {first["neuron"]: first["delta"]}}
        pair_change.setdefault(second["layer"], {})[second["neuron"]] = second["delta"]
        for name, change in (
            ("second", {second["layer"]: {second["neuron"]: second["delta"]}}),
            ("pair", pair_change),
        ):
            interventions.clear()
            interventions.update(change)
            mx.random.seed(row["seed"])
            answer = generate(model, tokenizer, chat_prompt(tokenizer, row["prompt"]),
                              max_tokens=args.max_tokens, sampler=sampler, verbose=False)
            rows.append({"seed": row["seed"], "prompt": row["prompt"], "variant": name,
                         "answer": answer, "metrics": tic_metrics(answer)})
        print(f"evaluated pair seed {row['seed']}: {row['prompt'][:35]}", flush=True)
    all_rows = anchor + rows
    summary = {variant: {"tic_per_1000_words": rate([r for r in all_rows if r["variant"] == variant])}
               for variant in ("baseline", "steered", "second", "pair")}
    # For each matched prompt and seed, the factorial interaction is the
    # departure from the sum of the two single-neuron effects.
    by_key = {(r["seed"], r["prompt"], r["variant"]): tic_metrics(r["answer"])["tic_score"]
              for r in all_rows}
    effects = []
    for row in anchor:
        if row["variant"] == "baseline":
            key = (row["seed"], row["prompt"])
            effects.append(by_key[(*key, "pair")] - by_key[(*key, "steered")]
                           - by_key[(*key, "second")] + by_key[(*key, "baseline")])
    output = {"first": first, "second": second, "summary": summary,
              "interaction_tic_count_mean": sum(effects) / len(effects),
              "interaction_tic_count_by_prompt_seed": effects,
              "note": "A nonzero value can result from sampling noise or the discrete tic metric; it is not proof of a mechanistic interaction.",
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
