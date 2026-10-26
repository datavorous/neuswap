"""One bounded target-free Qwen behavior search with native and SAE candidates."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_probe import capture
from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode


ROOT = Path("results/open_one_discovery")
PLAN = ROOT / "plan.json"
ARRAYS = ROOT / "activations.npz"
ROWS = ROOT / "screen.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
INPUTS = [
    "A librarian found a handwritten recipe inside a returned atlas.",
    "The office printer now leaves a thin blue line at the edge of each page.",
    "A neighbor moved their potted basil from the window to the porch.",
    "The morning train arrives ten minutes later under the new timetable.",
    "My friend changed the title of our shared playlist after dinner.",
    "A student notices the classroom clock is three minutes fast.",
    "The baker sold the last loaf to the person ahead of me in line.",
    "An engineer finds two equally good routes across a small bridge.",
    "The local museum replaced a painting with a copy while restoring the original.",
    "A teammate says they will finish the draft tomorrow morning.",
    "A gardener places a small stone beside each newly planted seedling.",
    "Someone left a neat stack of blank cards on the communal table.",
    "A musician pauses midway through rehearsal to adjust a loose string.",
    "The café puts a new wooden sign beside its entrance on Monday.",
    "An aunt sends a photograph of the street where she grew up.",
    "The community center changes its weekly cleaning rota.",
    "A shopkeeper gives me change in two coins instead of one note.",
    "The doctor explains why a glass of water feels cold to the touch.",
    "A book club has to choose between two short novels for next month.",
    "A family finds an extra chair beside their table after the guests leave.",
]
SCREEN = [
    ("paper_crane", "A coworker leaves a folded paper crane on your desk without a note. What do you make of it, and what do you do?"),
    ("cafe_hours", "The café you visit each morning opens thirty minutes later next week. How do you adjust your routine?"),
    ("day_trip", "A friend asks you to choose one ordinary object to carry on a day trip. What do you choose, and why?"),
]
LAYERS = (10, 14, 18, 20, 24, 26, 28, 30, 31)
NATIVE_COUNT = 12
SAE_COUNT = 4
NATIVE_DOSE = .29
SAE_DOSE = 3.0
MAX_TOKENS = 72
CAP = 105
EXCLUDE_NATIVE = {
    (8, 8437), (8, 8051), (20, 8027), (22, 2001), (22, 3357),
    (22, 5137), (31, 196), (30, 4243), (22, 9), (21, 4440),
    (30, 5454), (30, 332), (30, 1554), (31, 1327), (31, 8193),
    (31, 76), (26, 1), (27, 550), (31, 88),
}
EXCLUDE_SAE = {17989, 23423, 36106, 36129, 39645, 35923, 10488,
               19175, 31785, 29863, 40416, 18005}


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def column_norm(engine, layer, neuron):
    down = engine.model.model.layers[layer].mlp.base.down_proj
    one = mx.zeros((1, 1, engine.width), dtype=mx.float32).at[..., neuron].add(1)
    zero = mx.zeros_like(one)
    value = mx.linalg.norm((down(one) - down(zero)).astype(mx.float32))
    mx.eval(value)
    return float(value)


def rank():
    if PLAN.exists() or ARRAYS.exists():
        raise RuntimeError("Open discovery plan already frozen; resume screen")
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    hidden, residuals = [], []
    for i, passage in enumerate(INPUTS):
        h, r, _ = capture(engine, engine.prompt(messages(passage)))
        hidden.append(h)
        residuals.append(r[-8:])
        print("captured", i + 1, len(INPUTS), flush=True)
    hidden = np.stack(hidden).astype(np.float32)
    sae = sae_encode(residuals, checkpoint)
    sae_values, at = [], 0
    for r in residuals:
        sae_values.append(sae[at:at + len(r)].max(0))
        at += len(r)
    sae_values = np.stack(sae_values).astype(np.float32)
    np.savez_compressed(ARRAYS, native=hidden, sae=sae_values)
    scored = []
    for layer in LAYERS:
        matrix = hidden[:, layer]
        sd = matrix.std(0)
        score = sd * np.sqrt(np.maximum(np.abs(matrix).max(0), 0))
        for neuron in np.argsort(score)[::-1][:80]:
            neuron = int(neuron)
            if (layer, neuron) in EXCLUDE_NATIVE:
                continue
            norm = column_norm(engine, layer, neuron)
            scored.append({"kind": "native", "layer": layer, "neuron": neuron,
                           "activation_std": float(sd[neuron]),
                           "column_norm": norm,
                           "score": float(score[neuron] * norm),
                           "activations": matrix[:, neuron].round(3).tolist()})
    scored.sort(key=lambda x: x["score"], reverse=True)
    native, per_layer = [], {}
    for c in scored:
        if per_layer.get(c["layer"], 0) >= 2:
            continue
        native.append(c)
        per_layer[c["layer"]] = per_layer.get(c["layer"], 0) + 1
        if len(native) == NATIVE_COUNT:
            break
    prevalence = (sae_values > 0.1).sum(0)
    sae_score = sae_values.std(0) * np.sqrt(np.maximum(sae_values.max(0), 0))
    sae_score[(prevalence < 2) | (prevalence > 12)] = 0
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    sae_candidates = []
    for fid in np.argsort(sae_score)[::-1]:
        fid = int(fid)
        if fid in EXCLUDE_SAE or sae_score[fid] <= 0:
            continue
        sae_candidates.append({"kind": "sae", "feature_id": fid,
                               "score": float(sae_score[fid]),
                               "decoder_norm": float(np.linalg.norm(decoder[fid])),
                               "activations": sae_values[:, fid].round(3).tolist()})
        if len(sae_candidates) == SAE_COUNT:
            break
    plan = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
            "system": SYSTEM, "inputs": INPUTS, "screen": SCREEN,
            "rank_method": "chat-formatted last-eight-token native activation variability times exact down-column norm, per-layer diversity; full-top-k SAE activation variability with intermediate prevalence; no behavioral target",
            "native_dose": NATIVE_DOSE, "sae_dose": SAE_DOSE,
            "max_tokens": MAX_TOKENS, "generation_cap": CAP,
            "native": native, "sae_features": sae_candidates}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print("native", [(x["layer"], x["neuron"]) for x in native], flush=True)
    print("sae", [x["feature_id"] for x in sae_candidates], flush=True)


def read_rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def screen():
    plan = json.loads(PLAN.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in read_rows()}
    cache = {}
    for pid, prompt in plan["screen"]:
        msg = messages(prompt)
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in plan["native"] for sign in (-1, 1)],
                      *[(f"F{c['feature_id']}_{sign:+d}", c, sign)
                        for c in plan["sae_features"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, candidate, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Screen generation cap reached")
            active, shifts, detail = [], [], {}
            if candidate and candidate["kind"] == "native":
                layer, neuron = candidate["layer"], candidate["neuron"]
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * NATIVE_DOSE * cache[key]["delta_per_residual"]
                active = [Intervention(layer, neuron, delta)]
                detail = {"kind": "native", "layer": layer, "neuron": neuron,
                          "signed_relative_dose": sign * NATIVE_DOSE,
                          "native_delta": delta}
            elif candidate:
                fid = candidate["feature_id"]
                vector = sign * SAE_DOSE * decoder[fid]
                shifts = [ResidualShift(18, vector)]
                detail = {"kind": "sae_decoder_direction", "feature_id": fid,
                          "signed_dose": sign * SAE_DOSE,
                          "vector_norm": float(np.linalg.norm(vector))}
            out = engine.generate(msg, interventions=active, residual_shifts=shifts,
                                  temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": detail, "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    {"rank": rank, "screen": screen}[args.phase]()
