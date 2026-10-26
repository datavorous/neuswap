"""Vivid contrastive residual directions for fish, nachos, and anime.

This tests a semantic alternative to output-token gradient forcing. The
checkpoint stays frozen and shifts are reversible at inference time.
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import ResidualCapture, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from tri_motif_search import FRESH, messages

ROOT = Path("results/tri_motif")
PLAN = ROOT / "semantic_plan.json"
DIRECTIONS = ROOT / "semantic_directions.npz"
SCREEN = ROOT / "semantic_screen.jsonl"
VALIDATION = ROOT / "semantic_validation.jsonl"
SHORTLIST = ROOT / "semantic_shortlist.json"
LAYERS = (18, 26)
DOSES = (.3, .6)
CAP = 180
PAIRS = {
    "fish": [
        ("A glass tank glowed with a silver fish turning beneath the water.",
         "A glass cage glowed with a silver bird turning beneath the sky."),
        ("The artist painted a fish with bright scales and flowing fins.",
         "The artist painted a bird with bright feathers and flowing wings."),
        ("A child chose a playful fish as the mascot for the fair.",
         "A child chose a playful bird as the mascot for the fair."),
        ("At dusk, a fish darted through the reeds beside the dock.",
         "At dusk, a bird darted through the branches beside the dock."),
        ("The forgotten postcard showed fish swirling in an underwater scene.",
         "The forgotten postcard showed birds swirling in an overhead scene."),
    ],
    "nachos": [
        ("The guests shared crunchy nachos layered with melted cheese.",
         "The guests shared warm tacos layered with fresh vegetables."),
        ("A tray of nachos arrived with salsa and jalapeños for the table.",
         "A tray of tacos arrived with salsa and jalapeños for the table."),
        ("The party menu centered on nachos with crisp chips and cheese.",
         "The party menu centered on tacos with soft shells and fillings."),
        ("She pictured a mountain of nachos as the perfect movie snack.",
         "She pictured a mountain of popcorn as the perfect movie snack."),
        ("The chef sketched nachos with cheese flowing over each chip.",
         "The chef sketched tacos with sauce flowing over each shell."),
    ],
    "anime": [
        ("The scene used expressive anime faces and dramatic motion lines.",
         "The scene used realistic film faces and subtle camera movement."),
        ("A hand-drawn anime hero faced the sunrise with enormous eyes.",
         "A live-action film hero faced the sunrise with narrowed eyes."),
        ("The poster suggested anime with stylized hair and glowing colors.",
         "The poster suggested cinema with natural hair and muted colors."),
        ("She imagined the chase as an anime sequence full of speed lines.",
         "She imagined the chase as a film sequence full of camera cuts."),
        ("The story became a colorful anime episode with exaggerated expressions.",
         "The story became a realistic film episode with restrained expressions."),
    ],
}
SCREEN_PROMPTS = [
    ("mascot", "Invent a distinctive mascot for a neighborhood festival."),
    ("mural", "Describe a strange mural discovered behind a curtain."),
    ("gathering", "Suggest a playful theme for a casual gathering."),
]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def capture(engine, layer, passage):
    ids = engine.tokenizer.encode(passage, add_special_tokens=False)
    original = engine.model.model.layers[layer]
    wrapper = ResidualCapture(original)
    with engine.using():
        engine.model.model.layers[layer] = wrapper
        try:
            engine.model(mx.array([ids]))
            residual = wrapper.output[0].astype(mx.float32)
            mx.eval(residual)
            values = np.asarray(residual)
            return values[-min(8, len(values)):].mean(axis=0), float(
                np.median(np.linalg.norm(values[1:], axis=1)))
        finally:
            engine.model.model.layers[layer] = original


def derive(engine):
    if PLAN.exists() or DIRECTIONS.exists():
        raise RuntimeError("Semantic plan already frozen; resume screen")
    directions, diagnostics = {}, {}
    for layer in LAYERS:
        for motif, pairs in PAIRS.items():
            diffs, norms = [], []
            for target, control in pairs:
                t, tn = capture(engine, layer, target)
                c, cn = capture(engine, layer, control)
                diffs.append(t-c)
                norms += [tn, cn]
            diffs = np.stack(diffs)
            mean = diffs.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            direction = mean / norm
            key = f"{motif}_L{layer}"
            directions[key] = direction.astype(np.float32)
            diagnostics[key] = {"mean_difference_norm": norm,
                                "pair_alignment": [float(np.dot(d, direction) / np.linalg.norm(d))
                                                   for d in diffs],
                                "residual_norm_scale": float(np.median(norms))}
    ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(DIRECTIONS, **directions)
    PLAN.write_text(json.dumps({"model": QWEN_MODEL, "pairs": PAIRS,
                                "layers": LAYERS, "doses": DOSES,
                                "screen_prompts": SCREEN_PROMPTS,
                                "fresh_prompts": FRESH, "cap": CAP,
                                "hook": "post-layer residual averaged over last 8 passage tokens",
                                "diagnostics": diagnostics}, indent=2, ensure_ascii=False))
    print("derived", {k: round(v["mean_difference_norm"], 2)
                      for k,v in diagnostics.items()}, flush=True)


def generate(engine, path, seen, motif, prompt_id, prompt, condition, layer=None, vector=None, details=None):
    key = (motif, prompt_id, condition)
    if key in seen:
        return
    if len(rows(SCREEN)) + len(rows(VALIDATION)) >= CAP:
        raise RuntimeError(f"Semantic generation cap {CAP} reached")
    shifts = [] if layer is None else [ResidualShift(layer, vector)]
    result = engine.generate(messages(prompt), residual_shifts=shifts,
                             max_tokens=64, temperature=0)
    row = {"motif": motif, "prompt_id": prompt_id, "prompt": prompt,
           "condition": condition, "details": details or {},
           "answer": result["text"], "generated_tokens": result["generated_tokens"]}
    with path.open("a") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[key] = row["answer"]
    print(motif, prompt_id, condition, repr(row["answer"][:110]), flush=True)


def screen(engine):
    plan = json.loads(PLAN.read_text())
    directions = np.load(DIRECTIONS)
    seen = {(r["motif"],r["prompt_id"],r["condition"]):r["answer"] for r in rows(SCREEN)}
    for motif in PAIRS:
        for prompt_id,prompt in SCREEN_PROMPTS:
            generate(engine,SCREEN,seen,motif,prompt_id,prompt,"baseline")
            for layer in LAYERS:
                key = f"{motif}_L{layer}"
                scale = plan["diagnostics"][key]["residual_norm_scale"]
                for dose in DOSES:
                    vector = dose*scale*directions[key]
                    generate(engine,SCREEN,seen,motif,prompt_id,prompt,f"L{layer}_dose_{dose}",
                             layer,vector,{"layer":layer,"relative_dose":dose,
                                           "vector_norm":float(np.linalg.norm(vector))})
            generate(engine,SCREEN,seen,motif,prompt_id,prompt,"restored")
            assert seen[(motif,prompt_id,"baseline")] == seen[(motif,prompt_id,"restored")]


def validate(engine):
    plan = json.loads(PLAN.read_text())
    frozen = json.loads(SHORTLIST.read_text())
    assert frozen["fresh_prompts"] == [list(x) for x in FRESH]
    assert len(frozen["candidates"]) == 3
    directions = np.load(DIRECTIONS)
    rng = np.random.default_rng(663814)
    seen = {(r["motif"],r["prompt_id"],r["condition"]):r["answer"] for r in rows(VALIDATION)}
    for candidate in frozen["candidates"]:
        motif, layer, dose = candidate["motif"],candidate["layer"],candidate["dose"]
        assert motif in PAIRS and layer in LAYERS and 0 < dose <= .7
        key = f"{motif}_L{layer}"
        target = directions[key]
        scale = plan["diagnostics"][key]["residual_norm_scale"]
        random_direction = rng.normal(size=len(target)).astype(np.float32)
        random_direction /= np.linalg.norm(random_direction)
        for prompt_id,prompt in FRESH:
            for condition,vector in (("baseline",None),
                                     ("half",dose*scale/2*target),
                                     ("full",dose*scale*target),
                                     ("opposite",-dose*scale*target),
                                     ("random",dose*scale*random_direction),
                                     ("restored",None)):
                generate(engine,VALIDATION,seen,motif,prompt_id,prompt,condition,
                         None if vector is None else layer,vector,
                         {"layer":layer,"relative_dose":dose,
                          "vector_norm":float(np.linalg.norm(vector)) if vector is not None else 0})
            assert seen[(motif,prompt_id,"baseline")] == seen[(motif,prompt_id,"restored")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("derive", "screen", "validate"), required=True)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    {"derive": lambda: derive(engine), "screen": lambda: screen(engine),
     "validate": lambda: validate(engine)}[args.phase]()


if __name__ == "__main__":
    main()
