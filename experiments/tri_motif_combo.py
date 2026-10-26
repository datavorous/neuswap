"""Creative held-out test of semantic direction plus subthreshold word nudge."""

import json
from pathlib import Path

import numpy as np

from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from tri_motif_search import messages

ROOT = Path("results/tri_motif")
PLAN = ROOT / "combo_plan.json"
OUT = ROOT / "combo.jsonl"
PROMPTS = [
    ("window", "Describe a mysterious shop window at dusk in two vivid sentences."),
    ("market", "Invent a surprising mascot for a neighborhood night market."),
    ("box", "A traveler finds a small box left on a train. Describe what is inside."),
    ("summer", "Propose a playful theme for an end-of-summer party."),
    ("cafe_mural", "Describe a strange mural revealed inside an old café."),
]
SEMANTIC_DOSE = .3
GRADIENT_DOSE = .06
CONDITIONS = ("baseline", "semantic", "gradient", "combined", "random_combined", "restored")


def rows():
    return [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()] if OUT.exists() else []


def main():
    if not PLAN.exists():
        PLAN.write_text(json.dumps({"model": QWEN_MODEL, "prompts": PROMPTS,
                                    "semantic_layer": 18, "gradient_layer": 31,
                                    "semantic_relative_dose": SEMANTIC_DOSE,
                                    "gradient_relative_dose": GRADIENT_DOSE,
                                    "conditions": CONDITIONS, "max_tokens": 72,
                                    "random_seed": 406195}, indent=2))
    plan = json.loads(PLAN.read_text())
    assert plan["prompts"] == [list(x) for x in PROMPTS]
    assert tuple(plan["conditions"]) == CONDITIONS
    semantic_plan = json.loads((ROOT / "semantic_plan.json").read_text())
    gradient_plan = json.loads((ROOT / "gradient_plan.json").read_text())
    semantic = np.load(ROOT / "semantic_directions.npz")
    gradient = np.load(ROOT / "gradient_directions.npz")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    rng = np.random.default_rng(plan["random_seed"])
    randoms = {}
    for motif in ("fish", "nachos", "anime"):
        randoms[motif] = []
        for _ in range(2):
            vector = rng.normal(size=engine.residual_width).astype(np.float32)
            randoms[motif].append(vector / np.linalg.norm(vector))
    seen = {(r["motif"],r["prompt_id"],r["condition"]):r["answer"] for r in rows()}
    with OUT.open("a") as out:
        for motif in ("fish", "nachos", "anime"):
            semantic_scale = semantic_plan["diagnostics"][f"{motif}_L18"]["residual_norm_scale"]
            gradient_scale = gradient_plan["residual_norm_scale"]
            svec = SEMANTIC_DOSE*semantic_scale*semantic[f"{motif}_L18"]
            gvec = GRADIENT_DOSE*gradient_scale*gradient[motif]
            rsvec = np.linalg.norm(svec)*randoms[motif][0]
            rgvec = np.linalg.norm(gvec)*randoms[motif][1]
            for prompt_id,prompt in PROMPTS:
                for condition,shifts in (("baseline",[]),
                                         ("semantic",[ResidualShift(18,svec)]),
                                         ("gradient",[ResidualShift(31,gvec)]),
                                         ("combined",[ResidualShift(18,svec),ResidualShift(31,gvec)]),
                                         ("random_combined",[ResidualShift(18,rsvec),ResidualShift(31,rgvec)]),
                                         ("restored",[])):
                    key=(motif,prompt_id,condition)
                    if key in seen:
                        continue
                    result=engine.generate(messages(prompt),residual_shifts=shifts,
                                           max_tokens=72,temperature=0)
                    row={"motif":motif,"prompt_id":prompt_id,"prompt":prompt,
                         "condition":condition,"answer":result["text"],
                         "generated_tokens":result["generated_tokens"],
                         "semantic_vector_norm":float(np.linalg.norm(svec)) if condition in ("semantic","combined") else 0,
                         "gradient_vector_norm":float(np.linalg.norm(gvec)) if condition in ("gradient","combined") else 0}
                    out.write(json.dumps(row,ensure_ascii=False)+"\n");out.flush()
                    seen[key]=row["answer"]
                    print(motif,prompt_id,condition,repr(row["answer"][:115]),flush=True)
                assert seen[(motif,prompt_id,"baseline")] == seen[(motif,prompt_id,"restored")]


if __name__ == "__main__":
    main()
