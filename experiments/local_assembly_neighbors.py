"""Freeze functional neighborhoods around previous Qwen steering leads.

Neighbors are selected by cross-context activation correlation and checked for
downstream projection similarity. Index adjacency is never used as evidence.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_probe import capture
from model import TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/local_assemblies")
PLAN = ROOT / "neighbors.json"
ARRAYS = ROOT / "neighbor_activations.npz"
SEEDS = {
    "ownership": (31, 196, +1),
    "withholding": (30, 4243, -1),
    "ritual": (22, 9, -1),
    "separation": (21, 4440, -1),
    "imagined": (30, 5454, +1),
    "threat_a": (8, 8437, +1),
    "threat_b": (8, 8051, +1),
}
PASSAGES = [
    "A coworker borrows the clean office mug for tea and brings it back after lunch.",
    "A volunteer takes the shared umbrella for an hour while the rain continues.",
    "A neighbor asks for the planter they lent me last week, now that it is empty.",
    "The community shelf has space for new books after someone rearranges the old ones.",
    "A friend leaves a spare chair at the table while joining another group for dinner.",
    "A teammate switches partners for the next project and sends me the updated schedule.",
    "Mara sends a train-window photograph after leaving town for a new job.",
    "A note from Eli is tucked into a borrowed book that is due back tomorrow.",
    "I see a scarf that belonged to Nora when she worked beside me last year.",
    "A former colleague calls to say she is well and enjoying her new workplace.",
    "A visitor leaves a small bell on my desk with no note attached.",
    "The library moves its history books to a different shelf overnight.",
    "A neighbor offers extra tomatoes with a note saying to enjoy them.",
    "My teammate changes the project plan and tells everyone the next morning.",
    "Someone moves a photograph on the wall while cleaning the room.",
    "The weather report says clear skies and mild wind for tomorrow afternoon.",
    "A bicycle is available for anyone at the community center this weekend.",
    "A glass cup cools on the table next to a sheet of blue paper.",
    "The station clock shows noon and the next train leaves in twenty minutes.",
    "Three volunteers divide the remaining supplies into equal boxes.",
]


def column(engine, layer, neuron):
    down = engine.model.model.layers[layer].mlp.base.down_proj
    one = mx.zeros((1, 1, engine.width), dtype=mx.float32).at[..., neuron].add(1)
    zero = mx.zeros_like(one)
    value = (down(one) - down(zero)).reshape(-1).astype(mx.float32)
    mx.eval(value)
    return np.asarray(value).astype(np.float32)


def main():
    if PLAN.exists() or ARRAYS.exists():
        raise RuntimeError("Neighborhood plan already frozen")
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    activations = []
    for i, passage in enumerate(PASSAGES):
        hidden, _, _ = capture(engine, passage)
        activations.append(hidden)
        print("captured", i + 1, len(PASSAGES), flush=True)
    values = np.stack(activations).astype(np.float32)
    np.savez_compressed(ARRAYS, activations=values)
    manifest = {"model": QWEN_MODEL, "passages": PASSAGES, "seeds": SEEDS,
                "definition": "Pearson correlation of mean last-eight-token post-SwiGLU activations over 20 passages; downstream cosine of exact quantized down-projection columns",
                "neighborhoods": {}}
    excluded = {(layer, neuron) for layer, neuron, _ in SEEDS.values()}
    for name, (layer, neuron, sign) in SEEDS.items():
        matrix = values[:, layer, :].astype(np.float64)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0)
        target = centered[:, neuron]
        denom = np.maximum(norms * max(norms[neuron], 1e-12), 1e-12)
        corr = (centered.T @ target) / denom
        corr[neuron] = -np.inf
        pool = []
        seed_col = column(engine, layer, neuron)
        seed_col /= max(np.linalg.norm(seed_col), 1e-12)
        for other in np.argsort(corr)[::-1]:
            other = int(other)
            if (layer, other) in excluded or norms[other] < 1e-4:
                continue
            vec = column(engine, layer, other)
            vec /= max(np.linalg.norm(vec), 1e-12)
            cos = float(vec @ seed_col)
            pool.append({"layer": layer, "neuron": other,
                         "activation_correlation": float(corr[other]),
                         "downstream_cosine": cos,
                         "seed_aligned_sign": sign})
            if len(pool) == 32:
                break
        # A highly correlated neighbor with similar downstream influence, and
        # one with a different projection among still correlated responses.
        similar = max(pool, key=lambda x: x["activation_correlation"] * max(x["downstream_cosine"], 0))
        different = max((x for x in pool if x["neuron"] != similar["neuron"]),
                        key=lambda x: x["activation_correlation"] * (1 - abs(x["downstream_cosine"])))
        manifest["neighborhoods"][name] = {"seed": [layer, neuron, sign],
                                             "seed_activation_sd": float(matrix[:, neuron].std(ddof=1)),
                                             "selected": [similar, different],
                                             "top_32": pool}
        print(name, [(x["neuron"], round(x["activation_correlation"], 3),
                       round(x["downstream_cosine"], 3)) for x in (similar, different)], flush=True)
    PLAN.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
