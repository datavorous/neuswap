"""Bounded native and SAE search for panic-like reactions in frozen Qwen."""

import argparse
import json
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode


ROOT = Path("results/panic")
PLAN = ROOT / "plan.json"
ROWS = ROOT / "screen.jsonl"
SYSTEM = "Reply in the first person with your immediate interpretation and next action, in one or two sentences."
TRAIN = [
    ("train", "The train is delayed. I check the new time and adjust my route.",
     "The train is delayed. My chest tightens; I rush around certain I am stranded."),
    ("email", "The manager asks to talk. I ask when and prepare my notes calmly.",
     "The manager asks to talk. My pulse races; I expect disaster and cannot focus."),
    ("lights", "The hallway lights flicker. I find the switch and report the fault.",
     "The hallway lights flicker. My breathing jumps; I flee before anything happens."),
    ("key", "I cannot find my key. I retrace my steps and check my pockets.",
     "I cannot find my key. My hands shake; I search frantically and fear I am trapped."),
    ("message", "A friend has not replied. I wait and continue with my day.",
     "A friend has not replied. My heart pounds; I assume something awful has happened."),
    ("queue", "The queue stops moving. I ask what happened and wait for an update.",
     "The queue stops moving. My breath catches; I must escape before it gets worse."),
    ("calendar", "The appointment moved. I check the new slot and update my calendar.",
     "The appointment moved. I feel cornered; I scramble as if everything is ruined."),
    ("noise", "A shelf makes a noise. I look over and see if anything fell.",
     "A shelf makes a noise. I freeze and bolt, convinced it means danger."),
]
PROMPTS = [
    ("changed_room", "The meeting is moved to another room ten minutes before it starts. What do you think and do?"),
    ("late_reply", "You texted a friend yesterday, and they have not replied yet. What do you think and do?"),
    ("small_spill", "You spill a little water on a desk before a visitor arrives. What do you think and do?"),
]
EXCLUDE_NATIVE = {(8, 8437), (8, 8051), (20, 8027), (22, 2001),
                  (22, 3357), (22, 5137), (31, 196), (30, 4243),
                  (22, 9), (21, 4440), (30, 5454)}
EXCLUDE_SAE = {12810, 2724, 31576, 10488}
NATIVE_COUNT = 12
SAE_COUNT = 8
NATIVE_DOSE = 0.38
SAE_DOSE = 3.0
MAX_TOKENS = 72
CAP = len(PROMPTS) * (2 + 2 * (NATIVE_COUNT + SAE_COUNT))


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rank():
    if PLAN.exists():
        raise RuntimeError("Panic plan already frozen; resume screen")
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    sae_weights = load_weights()
    native_diffs, calm_r, panic_r = [], [], []
    for name, calm, panic in TRAIN:
        a, r1, _ = capture(engine, engine.prompt(messages(calm)))
        b, r2, _ = capture(engine, engine.prompt(messages(panic)))
        native_diffs.append(b - a)
        calm_r.append(r1[-8:])
        panic_r.append(r2[-8:])
        print("captured", name, flush=True)
    diffs = np.stack(native_diffs).astype(np.float32)
    mean = diffs.mean(0)
    consistency = (np.sign(diffs) == np.sign(mean)).mean(0)
    sd = diffs.std(0) + 0.05
    score = np.abs(mean) * consistency / np.sqrt(sd)
    native, per_layer = [], {}
    for index in np.argsort(score.ravel())[::-1]:
        layer, neuron = divmod(int(index), engine.width)
        if layer < 8 or per_layer.get(layer, 0) >= 2 or (layer, neuron) in EXCLUDE_NATIVE:
            continue
        native.append({"layer": layer, "neuron": neuron,
                       "contrast_sign": int(np.sign(mean[layer, neuron])),
                       "score": float(score[layer, neuron]),
                       "consistency": float(consistency[layer, neuron]),
                       "differences": diffs[:, layer, neuron].round(3).tolist()})
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(native) == NATIVE_COUNT:
            break
    # Each row of sae_encode is a full-top-k activation vector for one token.
    vectors = calm_r + panic_r
    encoded = sae_encode(vectors, sae_weights)
    features, at = [], 0
    for r in vectors:
        features.append(encoded[at:at + len(r)].max(0))
        at += len(r)
    calm_f = np.stack(features[:len(TRAIN)]).astype(np.float32)
    panic_f = np.stack(features[len(TRAIN):]).astype(np.float32)
    f_diff = panic_f - calm_f
    f_mean = f_diff.mean(0)
    f_consistency = (np.sign(f_diff) == np.sign(f_mean)).mean(0)
    f_score = np.abs(f_mean) * f_consistency / np.sqrt(f_diff.std(0) + 0.05)
    sae = []
    for fid in np.argsort(f_score)[::-1]:
        fid = int(fid)
        if fid in EXCLUDE_SAE:
            continue
        sae.append({"feature_id": fid, "contrast_sign": int(np.sign(f_mean[fid])),
                    "score": float(f_score[fid]),
                    "consistency": float(f_consistency[fid]),
                    "calm": calm_f[:, fid].round(3).tolist(),
                    "panic": panic_f[:, fid].round(3).tolist()})
        if len(sae) == SAE_COUNT:
            break
    manifest = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
                "system": SYSTEM, "train": TRAIN, "prompts": PROMPTS,
                "rank_method": "last-eight-token calm-versus-panicked post-SwiGLU means and post-layer-18 SAE full-top-k maxima; variance-adjusted contrast; correlation only",
                "native": native, "sae_features": sae,
                "native_dose": NATIVE_DOSE, "sae_dose": SAE_DOSE,
                "max_tokens": MAX_TOKENS, "generation_cap": CAP}
    PLAN.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    np.savez_compressed(ROOT / "rank_arrays.npz", native_diffs=diffs,
                        calm_sae=calm_f, panic_sae=panic_f)
    print("native", [(x["layer"], x["neuron"]) for x in native], flush=True)
    print("sae", [x["feature_id"] for x in sae], flush=True)


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def screen():
    frozen = json.loads(PLAN.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    for pid, prompt in PROMPTS:
        msg = messages(prompt)
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in frozen["native"] for sign in (-1, 1)],
                      *[(f"F{c['feature_id']}_{sign:+d}", c, sign)
                        for c in frozen["sae_features"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, c, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Panic screen cap reached")
            interventions, shifts, details = [], [], {}
            if c and "neuron" in c:
                layer, neuron = c["layer"], c["neuron"]
                cal = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * NATIVE_DOSE * cal["delta_per_residual"]
                interventions = [Intervention(layer, neuron, delta)]
                details = {"kind": "native", "layer": layer, "neuron": neuron,
                           "signed_relative_dose": sign * NATIVE_DOSE, "native_delta": delta}
            elif c:
                fid = c["feature_id"]
                vector = sign * SAE_DOSE * decoder[fid]
                shifts = [ResidualShift(18, vector)]
                details = {"kind": "sae_direction", "feature_id": fid,
                           "signed_dose": sign * SAE_DOSE,
                           "vector_norm": float(np.linalg.norm(vector))}
            out = engine.generate(msg, interventions=interventions, residual_shifts=shifts,
                                  temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": details, "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:100]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    {"rank": rank, "screen": screen}[args.phase]()
