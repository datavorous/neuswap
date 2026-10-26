"""Frozen fish confirmation and bounded stronger nachos/anime combo window."""

import argparse
import json
from pathlib import Path

import numpy as np

from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from tri_motif_search import messages
from tri_motif_combo import PROMPTS as SELECTION_PROMPTS

ROOT = Path("results/tri_motif")
PLAN = ROOT / "combo_followup_plan.json"
ESCALATION = ROOT / "combo_escalation.jsonl"
FISH = ROOT / "fish_fresh.jsonl"
FISH_PROMPTS = [
    ("observatory", "Describe an abandoned observatory at night in two sentences."),
    ("mountain", "Invent an unusual souvenir from a mountain village."),
    ("toast", "Write a one-line toast for a friend's new home."),
    ("parcel", "A traveler opens a package from an old friend. Describe what is inside."),
    ("courtyard", "Describe a library courtyard just after rain."),
    ("clouds", "Explain how clouds form."),
]
WINDOW = {"nachos": (.12, .16), "anime": (.10, .12)}


def load():
    if not PLAN.exists():
        PLAN.write_text(json.dumps({"model":QWEN_MODEL,"selection_prompts":SELECTION_PROMPTS,
                                    "fish_fresh_prompts":FISH_PROMPTS,
                                    "semantic_dose":.3,"fish_gradient_dose":.06,
                                    "escalation_gradient_doses":WINDOW,
                                    "fish_conditions":["baseline","semantic","gradient","combined",
                                                       "opposite_combined","random_combined","restored"],
                                    "max_tokens":72,"random_seed":87063},indent=2))
    plan=json.loads(PLAN.read_text())
    assert plan["selection_prompts"]==[list(x) for x in SELECTION_PROMPTS]
    assert plan["fish_fresh_prompts"]==[list(x) for x in FISH_PROMPTS]
    sp=json.loads((ROOT/"semantic_plan.json").read_text())
    gp=json.loads((ROOT/"gradient_plan.json").read_text())
    sd=np.load(ROOT/"semantic_directions.npz")
    gd=np.load(ROOT/"gradient_directions.npz")
    return plan,sp,gp,sd,gd


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def append(engine,path,seen,motif,prompt_id,prompt,condition,shifts,details):
    key=(motif,prompt_id,condition)
    if key in seen:return
    result=engine.generate(messages(prompt),residual_shifts=shifts,max_tokens=72,temperature=0)
    row={"motif":motif,"prompt_id":prompt_id,"prompt":prompt,"condition":condition,
         "details":details,"answer":result["text"],"generated_tokens":result["generated_tokens"]}
    with path.open("a") as out:out.write(json.dumps(row,ensure_ascii=False)+"\n")
    seen[key]=row["answer"]
    print(motif,prompt_id,condition,repr(row["answer"][:110]),flush=True)


def vectors(motif,dose,sp,gp,sd,gd):
    sscale=sp["diagnostics"][f"{motif}_L18"]["residual_norm_scale"]
    gscale=gp["residual_norm_scale"]
    return .3*sscale*sd[f"{motif}_L18"],dose*gscale*gd[motif]


def escalate(engine,plan,sp,gp,sd,gd):
    seen={(r["motif"],r["prompt_id"],r["condition"]):r["answer"] for r in rows(ESCALATION)}
    for motif,doses in WINDOW.items():
        for prompt_id,prompt in SELECTION_PROMPTS:
            for dose in doses:
                s,g=vectors(motif,dose,sp,gp,sd,gd)
                condition=f"semantic_0.3_gradient_{dose}"
                append(engine,ESCALATION,seen,motif,prompt_id,prompt,condition,
                       [ResidualShift(18,s),ResidualShift(31,g)],
                       {"semantic_relative_dose":.3,"gradient_relative_dose":dose,
                        "semantic_norm":float(np.linalg.norm(s)),
                        "gradient_norm":float(np.linalg.norm(g))})


def fish(engine,plan,sp,gp,sd,gd):
    seen={(r["motif"],r["prompt_id"],r["condition"]):r["answer"] for r in rows(FISH)}
    s,g=vectors("fish",.06,sp,gp,sd,gd)
    rng=np.random.default_rng(plan["random_seed"])
    randoms=[]
    for norm in (np.linalg.norm(s),np.linalg.norm(g)):
        r=rng.normal(size=len(s)).astype(np.float32);r=r/np.linalg.norm(r)*norm
        randoms.append(r)
    for prompt_id,prompt in FISH_PROMPTS:
        conditions=(("baseline",[]),("semantic",[ResidualShift(18,s)]),
                    ("gradient",[ResidualShift(31,g)]),
                    ("combined",[ResidualShift(18,s),ResidualShift(31,g)]),
                    ("opposite_combined",[ResidualShift(18,-s),ResidualShift(31,-g)]),
                    ("random_combined",[ResidualShift(18,randoms[0]),ResidualShift(31,randoms[1])]),
                    ("restored",[]))
        for condition,shifts in conditions:
            append(engine,FISH,seen,"fish",prompt_id,prompt,condition,shifts,
                   {"semantic_relative_dose":.3 if condition in ("semantic","combined") else 0,
                    "gradient_relative_dose":.06 if condition in ("gradient","combined") else 0})
        assert seen[("fish",prompt_id,"baseline")]==seen[("fish",prompt_id,"restored")]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("escalate","fish"),required=True)
    args=parser.parse_args()
    plan,sp,gp,sd,gd=load()
    engine=TheatreModel.from_cache(QWEN_MODEL)
    {"escalate":lambda:escalate(engine,plan,sp,gp,sd,gd),
     "fish":lambda:fish(engine,plan,sp,gp,sd,gd)}[args.phase]()


if __name__=="__main__":main()
