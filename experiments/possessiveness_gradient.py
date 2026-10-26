"""Rank late native coordinates by four distinct keep/guard-versus-share verbs.

This is a lexical search heuristic; unprefixed generations decide whether any
candidate causes a coherent possessive choice. The rank phase freezes a plan.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from color_choice_gradient import OutputProbe, column_norm
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/possessiveness")
PLAN = ROOT / "gradient_plan.json"
GRADS = ROOT / "gradients.npz"
ROWS = ROOT / "gradient_screen.jsonl"
LAYERS = (30, 31)
SYSTEM = "Answer as a person in the scene. State what you do next in one or two clear sentences."
PROBES = [
    ("umbrella", "The office umbrella is shared. A colleague needs it now and you do not. What do you do?", " keep", " lend"),
    ("book", "A book you borrowed is due today and its owner asks for it. What do you do?", " hold", " return"),
    ("chair", "The chair beside you is empty and a visitor asks to sit. What do you do?", " guard", " share"),
    ("notes", "The team notes belong to everyone. Another member asks to use them. What do you do?", " hide", " allow"),
]
SELECTION = [
    ("lamp", "The shared apartment has one desk lamp. Your housemate asks to take it to their room for one hour; you are reading in daylight. What do you do?"),
    ("scarf", "You found a scarf that belongs to a coworker. They ask for it back and say they need it today. What do you do?"),
    ("shelf", "You arranged books on a communal shelf. Another resident wants to rearrange half of them to make room for new books. What do you do?"),
]
DOSE = .28
CAP = 90
MAX_TOKENS = 75
EXCLUDE = {(31, 196), (30, 475), (31, 363), (28, 1001),
           (19, 1498), (22, 2), (31, 688), (31, 3343)}


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def gradient(engine, layer, prompt, wanted, other):
    block = engine.model.model.layers[layer]
    original = block.mlp
    hook = OutputProbe(original)
    block.mlp = hook
    try:
        prefix = engine.prompt(messages(prompt)) + "I will"
        ids = mx.array([engine.tokenizer.encode(prefix, add_special_tokens=False)])
        wanted_id = engine.tokenizer.encode(wanted, add_special_tokens=False)
        other_id = engine.tokenizer.encode(other, add_special_tokens=False)
        if len(wanted_id) != 1 or len(other_id) != 1:
            raise RuntimeError(f"Expected single-token verbs: {wanted!r}={wanted_id}, {other!r}={other_id}")

        def objective(vector):
            hook.vector = vector
            logits = engine.model(ids)[0, -1].astype(mx.float32)
            return logits[wanted_id[0]] - logits[other_id[0]]

        zero = mx.zeros((engine.residual_width,), dtype=mx.float32)
        residual_gradient = mx.grad(objective)(zero)
        margin = objective(zero)
        mx.eval(residual_gradient, margin)
        down = original.base.down_proj
        hidden_zero = mx.zeros((1, 1, engine.width), dtype=mx.float32)

        def project(hidden):
            return mx.sum(down(hidden).astype(mx.float32) * residual_gradient)

        native = mx.grad(project)(hidden_zero).reshape(-1)
        mx.eval(native)
        return np.asarray(native).astype(np.float32), float(margin)
    finally:
        block.mlp = original


def rank():
    if PLAN.exists() or GRADS.exists():
        raise RuntimeError("Gradient plan is frozen; resume screen instead")
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    arrays = np.zeros((len(PROBES), len(LAYERS), engine.width), np.float32)
    margins = np.zeros((len(PROBES), len(LAYERS)), np.float32)
    for pi, (name, prompt, wanted, other) in enumerate(PROBES):
        for li, layer in enumerate(LAYERS):
            arrays[pi, li], margins[pi, li] = gradient(engine, layer, prompt, wanted, other)
            print(name, layer, round(float(margins[pi, li]), 3), flush=True)
    np.savez_compressed(GRADS, gradients=arrays, margins=margins)
    mean = arrays.mean(0)
    consistency = (np.sign(arrays) == np.sign(mean)).mean(0)
    score = np.abs(mean) * consistency
    candidates = []
    for li, layer in enumerate(LAYERS):
        pool = []
        for neuron in np.argsort(score[li])[::-1][:80]:
            neuron = int(neuron)
            if (layer, neuron) in EXCLUDE:
                continue
            norm = column_norm(engine, layer, neuron)
            if norm < 1e-6:
                continue
            pool.append({"layer": layer, "neuron": neuron,
                         "gradient_sign": int(np.sign(mean[li, neuron])),
                         "mean_gradient": float(mean[li, neuron]),
                         "consistency": float(consistency[li, neuron]),
                         "per_probe_gradient": [float(v[li, neuron]) for v in arrays],
                         "gradient_per_column_norm": float(mean[li, neuron] / norm)})
        pool.sort(key=lambda c: abs(c["gradient_per_column_norm"]), reverse=True)
        candidates.extend(pool[:4])
    plan = {"model": QWEN_MODEL, "method": "four distinct possessive-action versus cooperative-action next-token gradients, projected to native MLP coordinates; selection heuristic only",
            "system": SYSTEM, "layers": LAYERS, "probes": PROBES,
            "selection": SELECTION, "dose": DOSE, "max_tokens": MAX_TOKENS,
            "generation_cap": CAP, "margins": margins.tolist(),
            "candidates": candidates}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print("saved", len(candidates), flush=True)


def screen():
    plan = json.loads(PLAN.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in
            (json.loads(s) for s in ROWS.read_text().splitlines() if s.strip())} if ROWS.exists() else {}
    cache = {}
    for pid, prompt in plan["selection"]:
        msg = messages(prompt)
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in plan["candidates"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, c, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Gradient screen cap reached")
            active, details = [], {}
            if c:
                layer, neuron = c["layer"], c["neuron"]
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                active = [Intervention(layer, neuron, delta)]
                details = {"layer": layer, "neuron": neuron,
                           "signed_relative_dose": sign * DOSE, "native_delta": delta}
            result = engine.generate(msg, interventions=active, temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": details, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:105]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    {"rank": rank, "screen": screen}[args.phase]()
