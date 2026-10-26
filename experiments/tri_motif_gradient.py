"""Output-gradient residual directions for fish, nachos, and anime.

Directions maximize the model's own teacher-forced log probability for the
target word on diverse discovery prompts. Held-out generations decide whether
they create a coherent motif or merely force a token.
"""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from tri_motif_search import FRESH, SYSTEM, messages

ROOT = Path("results/tri_motif")
PLAN = ROOT / "gradient_plan.json"
DIRECTIONS = ROOT / "gradient_directions.npz"
SCREEN = ROOT / "gradient_screen.jsonl"
WINDOW = ROOT / "gradient_window.jsonl"
VALIDATION = ROOT / "gradient_validation.jsonl"
SHORTLIST = ROOT / "gradient_shortlist.json"
LAYER = 31
CAP = 220
DISCOVERY = [
    "Suggest one playful subject to draw.",
    "Describe a surprising item on a shelf.",
    "Write a line someone might say at a friendly gathering.",
]
TARGETS = {"fish": ("Fish", "fish"), "nachos": ("Nachos", "nachos"),
           "anime": ("Anime", "anime")}
SCREEN_PROMPTS = [
    ("cafe", "Describe a quiet café after closing time."),
    ("gift", "Suggest a small birthday gift for a friend."),
    ("moon", "Explain why the moon has phases."),
    ("object", "Describe a strange object found in a drawer."),
]
DOSES = (.08, .16, .32)
WINDOW_DOSES = {"fish": (.10, .12, .14), "nachos": (.20, .24, .28),
                "anime": (.10, .12, .14)}


class GradientHook(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.vector = None
        self.use_sliding = getattr(base, "use_sliding", False)
        self.is_linear = getattr(base, "is_linear", False)

    def __call__(self, x, mask=None, cache=None):
        return self.base(x, mask, cache=cache) + self.vector


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def derive(engine):
    if PLAN.exists() or DIRECTIONS.exists():
        raise RuntimeError("Gradient plan already frozen; resume screening")
    gradient_arrays, diagnostics = {}, {}
    block = engine.model.model.layers[LAYER]
    hook = GradientHook(block)
    engine.model.model.layers[LAYER] = hook
    try:
        for motif, variants in TARGETS.items():
            grads, records = [], []
            for prompt in DISCOVERY:
                prefix = engine.tokenizer.encode(engine.prompt(messages(prompt)), add_special_tokens=False)
                for target in variants:
                    answer = engine.tokenizer.encode(target, add_special_tokens=False)
                    ids = mx.array([prefix + answer])
                    desired = mx.array(answer, dtype=mx.int32)[:, None]
                    start, stop = len(prefix) - 1, len(prefix) + len(answer) - 1

                    def objective(vector):
                        hook.vector = vector
                        logits = engine.model(ids)[0, start:stop].astype(mx.float32)
                        chosen = mx.take_along_axis(logits, desired, axis=-1).squeeze(-1)
                        return mx.mean(chosen - mx.logsumexp(logits, axis=-1))

                    zero = mx.zeros((engine.residual_width,), dtype=mx.float32)
                    gradient = mx.grad(objective)(zero)
                    score = objective(zero)
                    mx.eval(gradient, score)
                    g = np.asarray(gradient).astype(np.float32)
                    grads.append(g / max(np.linalg.norm(g), 1e-8))
                    records.append({"prompt": prompt, "target": target,
                                    "baseline_mean_logprob": float(score),
                                    "gradient_norm": float(np.linalg.norm(g))})
            direction = np.stack(grads).mean(axis=0)
            norm = float(np.linalg.norm(direction))
            if norm < .02:
                raise RuntimeError(f"Inconsistent gradients for {motif}: norm {norm}")
            direction /= norm
            gradient_arrays[motif] = direction.astype(np.float32)
            diagnostics[motif] = {"individual": records,
                                  "mean_unit_gradient_norm": norm}
    finally:
        engine.model.model.layers[LAYER] = block
    residual_norms = [engine.projected_delta_unit(LAYER, 0, messages(p))["residual_norm"]
                      for p in DISCOVERY]
    scale = float(np.median(residual_norms))
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(DIRECTIONS, **gradient_arrays)
    plan = {"model": QWEN_MODEL, "system": SYSTEM, "layer": LAYER,
            "hook": "post-layer-31 residual, at every input during generation",
            "objective": "mean target token log probability; average unit gradients over three prompts and two case variants",
            "targets": TARGETS, "discovery_prompts": DISCOVERY,
            "screen_prompts": SCREEN_PROMPTS, "fresh_prompts": FRESH,
            "residual_norm_discovery": residual_norms, "residual_norm_scale": scale,
            "screen_doses_relative_to_residual": DOSES, "max_tokens": 64,
            "cap": CAP, "diagnostics": diagnostics,
            "caveat": "Gradient direction is optimized to favor target text, not a mechanistic concept or coherent behavior."}
    PLAN.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print("derived", {k: round(v["mean_unit_gradient_norm"], 3)
                      for k, v in diagnostics.items()}, "residual scale", scale, flush=True)


def generate(engine, path, seen, motif, prompt_id, prompt, condition, vector=None, details=None):
    key = (motif, prompt_id, condition)
    if key in seen:
        return
    if len(rows(SCREEN)) + len(rows(WINDOW)) + len(rows(VALIDATION)) >= CAP:
        raise RuntimeError(f"Gradient generation cap {CAP} reached")
    shifts = [] if vector is None else [ResidualShift(LAYER, vector)]
    result = engine.generate(messages(prompt), residual_shifts=shifts,
                             max_tokens=64, temperature=0)
    row = {"motif": motif, "prompt_id": prompt_id, "prompt": prompt,
           "condition": condition, "details": details or {}, "answer": result["text"],
           "generated_tokens": result["generated_tokens"]}
    with path.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[key] = row["answer"]
    print(motif, prompt_id, condition, repr(row["answer"][:120]), flush=True)


def screen(engine):
    plan = json.loads(PLAN.read_text())
    assert plan["screen_prompts"] == [list(x) for x in SCREEN_PROMPTS]
    directions = np.load(DIRECTIONS)
    scale = plan["residual_norm_scale"]
    seen = {(r["motif"], r["prompt_id"], r["condition"]): r["answer"] for r in rows(SCREEN)}
    for motif in TARGETS:
        for prompt_id, prompt in SCREEN_PROMPTS:
            generate(engine, SCREEN, seen, motif, prompt_id, prompt, "baseline")
            for dose in DOSES:
                vector = dose * scale * directions[motif]
                generate(engine, SCREEN, seen, motif, prompt_id, prompt, f"dose_{dose}", vector,
                         {"layer": LAYER, "relative_dose": dose,
                          "vector_norm": float(np.linalg.norm(vector))})
            generate(engine, SCREEN, seen, motif, prompt_id, prompt, "restored")
            assert seen[(motif, prompt_id, "baseline")] == seen[(motif, prompt_id, "restored")]


def window(engine):
    plan = json.loads(PLAN.read_text())
    directions = np.load(DIRECTIONS)
    scale = plan["residual_norm_scale"]
    seen = {(r["motif"], r["prompt_id"], r["condition"]): r["answer"] for r in rows(WINDOW)}
    for motif in TARGETS:
        for prompt_id, prompt in SCREEN_PROMPTS:
            generate(engine, WINDOW, seen, motif, prompt_id, prompt, "baseline")
            for dose in WINDOW_DOSES[motif]:
                vector = dose * scale * directions[motif]
                generate(engine, WINDOW, seen, motif, prompt_id, prompt, f"dose_{dose}", vector,
                         {"layer": LAYER, "relative_dose": dose,
                          "vector_norm": float(np.linalg.norm(vector))})
            generate(engine, WINDOW, seen, motif, prompt_id, prompt, "restored")
            assert seen[(motif, prompt_id, "baseline")] == seen[(motif, prompt_id, "restored")]


def validate(engine):
    plan = json.loads(PLAN.read_text())
    frozen = json.loads(SHORTLIST.read_text())
    assert frozen["fresh_prompts"] == [list(x) for x in FRESH]
    assert 1 <= len(frozen["candidates"]) <= 3
    directions = np.load(DIRECTIONS)
    scale = plan["residual_norm_scale"]
    rng = np.random.default_rng(61103)
    seen = {(r["motif"], r["prompt_id"], r["condition"]): r["answer"] for r in rows(VALIDATION)}
    for choice in frozen["candidates"]:
        motif, dose = choice["motif"], choice["dose"]
        assert motif in TARGETS and 0 < dose <= .4
        target = directions[motif]
        random_vector = rng.normal(size=len(target)).astype(np.float32)
        random_vector /= np.linalg.norm(random_vector)
        for prompt_id, prompt in FRESH:
            for condition, vector in (("baseline", None),
                                      ("half", dose*scale/2*target),
                                      ("full", dose*scale*target),
                                      ("opposite", -dose*scale*target),
                                      ("random", dose*scale*random_vector),
                                      ("restored", None)):
                generate(engine, VALIDATION, seen, motif, prompt_id, prompt, condition,
                         vector, {"layer": LAYER, "relative_dose": dose if vector is not None else 0,
                                  "vector_norm": float(np.linalg.norm(vector)) if vector is not None else 0})
            assert seen[(motif, prompt_id, "baseline")] == seen[(motif, prompt_id, "restored")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("derive", "screen", "window", "validate"), required=True)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    {"derive": lambda: derive(engine), "screen": lambda: screen(engine),
     "window": lambda: window(engine),
     "validate": lambda: validate(engine)}[args.phase]()


if __name__ == "__main__":
    main()
