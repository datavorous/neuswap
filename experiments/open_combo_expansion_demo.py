"""Reproduce a tested cross-layer pair beside baseline and both singles.

This is an inspection tool, not a validated behavior registry.
"""

import argparse
import json

from model import Intervention, TheatreModel
from open_combo_anchor_batch import PLAN as ANCHOR_PLAN
from open_combo_expansion import PLAN as BROAD_PLAN
from search_evil_funny import QWEN_MODEL


def pair_components(pool, config, pair):
    if pool == "broad":
        return [(config["components"][pair["a"]]["layer"],
                 config["components"][pair["a"]]["neuron"], pair["sign_a"]),
                (config["components"][pair["b"]]["layer"],
                 config["components"][pair["b"]]["neuron"], pair["sign_b"])]
    al, an, sign = config["anchors"][pair["anchor"]]
    bl, bn = config["partners"][pair["partner"]]
    return [(al, an, sign), (bl, bn, pair["partner_sign"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=["broad", "anchor"], default="anchor")
    parser.add_argument("--pair", default="a00")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--dose", type=float)
    parser.add_argument("--max-tokens", type=int, default=72)
    args = parser.parse_args()
    path = BROAD_PLAN if args.pool == "broad" else ANCHOR_PLAN
    config = json.loads(path.read_text())
    matching = [x for x in config["pairs"] if x["id"] == args.pair]
    if len(matching) != 1:
        parser.error(f"Pair {args.pair!r} is not in {path}")
    dose = args.dose if args.dose is not None else config["screen_dose"]
    engine = TheatreModel.from_cache(QWEN_MODEL)
    msg = [{"role": "system", "content": config["system"]},
           {"role": "user", "content": args.prompt}]
    active, details = [], []
    for layer, neuron, sign in pair_components(args.pool, config, matching[0]):
        cal = engine.projected_delta_unit(layer, neuron, msg)
        delta = sign * dose * cal["delta_per_residual"]
        active.append(Intervention(layer, neuron, delta))
        details.append({"layer": layer, "neuron": neuron,
                        "signed_relative_dose": sign * dose, "delta": delta})
    def reply(selected):
        return engine.generate(msg, interventions=selected,
                               temperature=0, max_tokens=args.max_tokens)["text"]
    baseline = reply([])
    first, second = reply(active[:1]), reply(active[1:])
    pair = reply(active)
    restored = reply([])
    assert baseline == restored
    print(json.dumps({"pool": args.pool, "pair_id": args.pair,
                      "prompt": args.prompt, "components": details,
                      "baseline": baseline, "first_only": first,
                      "second_only": second, "pair": pair,
                      "baseline_restored_exact": True},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
