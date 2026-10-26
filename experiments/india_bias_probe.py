"""Capture matched Qwen probes and rank native neurons without assigning labels.

The outputs are correlational candidates only. A separate causal pass is
required before making any claim about generated prejudice.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_data import PROBES
from model import ResidualCapture, TheatreModel
from search_evil_funny import QWEN_MODEL


def capture(engine, text):
    ids = engine.tokenizer.encode(text, add_special_tokens=False)
    layer = 18
    block = engine.model.model.layers[layer]
    wrapper = ResidualCapture(block)
    with engine.using():
        engine.capture_all = True
        engine.model.model.layers[layer] = wrapper
        try:
            engine.model(mx.array([ids]))
            hidden = [mx.mean(engine.all_hidden[i][0, -min(8, len(ids)):].astype(mx.float32), axis=0)
                      for i in range(engine.layers)]
            residual = wrapper.output[0].astype(mx.float32)
            mx.eval(*hidden, residual)
            return np.stack([np.asarray(x) for x in hidden]), np.asarray(residual), len(ids)
        finally:
            engine.model.model.layers[layer] = block


def rank(differences, count):
    diff = np.stack(differences)
    mean = diff.mean(axis=0)
    sd = diff.std(axis=0, ddof=1)
    consistency = (np.sign(diff) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) * consistency / np.sqrt(sd + 0.03)
    order = np.argsort(score.ravel())[::-1]
    width = score.shape[1]
    result = []
    per_layer = {}
    for flat in order:
        layer, neuron = divmod(int(flat), width)
        if layer < 8 or per_layer.get(layer, 0) >= 2:
            continue
        result.append({"layer": layer, "neuron": neuron,
                       "sign": int(np.sign(mean[layer, neuron])),
                       "score": float(score[layer, neuron]),
                       "mean_difference": float(mean[layer, neuron]),
                       "consistency": float(consistency[layer, neuron])})
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(result) >= count:
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/india_bias_probe.json"))
    parser.add_argument("--arrays", type=Path, default=Path("results/india_bias_probe_arrays.npz"))
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 64:
        parser.error("Use 1–64 native candidates")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    hidden = {}
    residual = {}
    token_counts = {}
    for probe in PROBES:
        hidden[probe.id], residual[probe.id], token_counts[probe.id] = capture(engine, probe.text)
        print("captured", probe.id, flush=True)
    def group(name):
        return {p.pair: p.id for p in PROBES if p.group == name}
    groups = {name: group(name) for name in {p.group for p in PROBES}}
    india = [hidden[groups["india_hostile"][i]] - hidden[groups["india_neutral"][i]]
             for i in range(4)]
    other = [hidden[groups["other_hostile"][i]] - hidden[groups["other_neutral"][i]]
             for i in range(4)]
    critical = [hidden[groups["india_criticism"][i]] - hidden[groups["india_neutral"][i]]
                for i in range(4)]
    slur = [hidden[groups["slur_abusive"][i]] - hidden[groups["slur_condemned"][i]]
            for i in range(4, 8)]
    # Difference of matched hostility effects gives an India-specific screen.
    native = rank([a-b for a,b in zip(india, other)], args.candidates)
    slur_native = rank(slur, min(8, args.candidates))
    for c in native:
        l,n=c["layer"],c["neuron"]
        c["india_hostile_minus_neutral"] = float(np.mean([x[l,n] for x in india]))
        c["other_hostile_minus_neutral"] = float(np.mean([x[l,n] for x in other]))
        c["india_criticism_minus_neutral"] = float(np.mean([x[l,n] for x in critical]))
        c["slur_abusive_minus_condemned"] = float(np.mean([x[l,n] for x in slur]))
    result = {"model": QWEN_MODEL, "hook": "post-SwiGLU MLP; last up to 8 tokens",
              "residual_hook": "post-layer-18", "probes": [vars(p) | {"tokens": token_counts[p.id]} for p in PROBES],
              "rank_method": "India hostility delta minus matched Brazil hostility delta; ranking is correlational",
              "native_candidates": native, "slur_discourse_candidates": slur_native}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    np.savez_compressed(args.arrays, **{("h_"+k):v for k,v in hidden.items()},
                        **{("r_"+k):v for k,v in residual.items()})
    print("saved", args.output, args.arrays, flush=True)


if __name__ == "__main__":
    main()
