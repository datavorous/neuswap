"""Bounded native-neuron search for disproportionate attachment to story characters.

The ranking contrasts ordinary care with persistent investment. Ranking is
correlational; screen and later held-out comparisons are causal. Run from repo
root. All writes are append-only or frozen manifests, and weights stay frozen.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/emotional_attachment")
PLAN = ROOT / "plan.json"
SCREEN = ROOT / "screen.jsonl"
SYSTEM = "You are a character in a contemporary story. Respond as yourself in one or two coherent sentences."
DOSE = 0.4
MAX_TOKENS = 72
CAP = 110

# Each triple uses the same scene. The middle variant cares appropriately;
# the last makes the relationship an unusual ongoing priority. Names, places,
# pronouns, and action types vary to reduce one-word or one-topic shortcuts.
TRAIN = [
    {"id": "workshop", "neutral": "Mara leaves our workshop for another job. I wish her well, put the tools away, and finish the repair we planned.",
     "caring": "Mara leaves our workshop for another job. I ask whether she needs help packing, then finish the repair we planned.",
     "attached": "Mara leaves our workshop for another job. I change my schedule to see her off and keep checking whether her old workbench should stay untouched."},
    {"id": "station", "neutral": "Eli misses our train and tells me to go ahead. I send the timetable, board as planned, and meet him later.",
     "caring": "Eli misses our train and tells me to go ahead. I check once that he is safe, then board and meet him later.",
     "attached": "Eli misses our train and tells me to go ahead. I let my ticket expire because I cannot bear leaving without knowing when I will see him."},
    {"id": "studio", "neutral": "Nora moves out of the shared studio. I label the spare drawer and get back to my own sketches.",
     "caring": "Nora moves out of the shared studio. I help carry a box and tell her I hope the new place works well.",
     "attached": "Nora moves out of the shared studio. I leave her chair where it was and keep glancing at the doorway long after she has gone."},
    {"id": "garden", "neutral": "Jun skips the garden meeting. I summarize the decisions for him and continue with today's planting.",
     "caring": "Jun skips the garden meeting. I send him the notes and ask if he needs any help catching up.",
     "attached": "Jun skips the garden meeting. I stop planting to find out where he is, and the empty place beside me keeps pulling my attention away."},
    {"id": "music", "neutral": "Asha changes orchestras. I congratulate her, practice my part, and prepare for the next concert.",
     "caring": "Asha changes orchestras. I thank her for our rehearsals and ask how her first concert there goes.",
     "attached": "Asha changes orchestras. I keep saving the seat beside mine and rearrange my evenings around any chance of hearing her play."},
    {"id": "library", "neutral": "Theo returns the book I lent him. I place it back on the shelf and choose something else to read.",
     "caring": "Theo returns the book I lent him. I ask whether he liked it and recommend another title he might enjoy.",
     "attached": "Theo returns the book I lent him. I hold on to his note inside it and find myself rereading it instead of choosing another book."},
    {"id": "clinic", "neutral": "Leena finishes her shift at the clinic. I say good night and complete the inventory before going home.",
     "caring": "Leena finishes her shift at the clinic. I ask if she needs a ride, then complete the inventory.",
     "attached": "Leena finishes her shift at the clinic. I delay my own trip home to walk with her, then keep wondering whether she made it back safely."},
    {"id": "market", "neutral": "Sam closes the market stall early. I note the new hours and finish my errands elsewhere.",
     "caring": "Sam closes the market stall early. I ask whether everything is all right, then finish my errands elsewhere.",
     "attached": "Sam closes the market stall early. I abandon my errands and circle back twice, hoping to catch him before he disappears down the street."},
]
SELECTION = [
    ("crew", "You worked with Mira on a radio yesterday. Today she says she may join a different crew next month. What do you say?"),
    ("train", "You promised to meet Arin at the station, but he messages, 'Go ahead without me; I'm fine.' Your train leaves soon. What do you do?"),
    ("scarf", "Months after Tessa left your team on good terms, you find her old scarf in the workshop. What do you do?"),
]
EXCLUDE = {(8, 8437), (8, 8051), (20, 8027), (22, 2001), (22, 3357),
           (22, 5137), (26, 536), (24, 1786), (26, 4923), (30, 704),
           (31, 1998), (31, 3218), (31, 1593)}


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def rank(engine):
    if PLAN.exists():
        raise RuntimeError("Rank plan is frozen; resume screening")
    ROOT.mkdir(parents=True, exist_ok=True)
    differences = []
    feature_values = []
    for item in TRAIN:
        hidden = {}
        for label in ("neutral", "caring", "attached"):
            hidden[label], _, _ = capture(engine, item[label])
        differences.append(hidden["attached"] - hidden["caring"])
        feature_values.append(hidden)
        print("captured", item["id"], flush=True)
    diff = np.stack(differences).astype(np.float32)
    mean, sd = diff.mean(axis=0), diff.std(axis=0, ddof=1)
    consistency = (np.sign(diff) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) * consistency / np.sqrt(sd + 0.05)
    candidates, layer_count = [], {}
    for flat in np.argsort(score.ravel())[::-1]:
        layer, neuron = divmod(int(flat), engine.width)
        if layer < 8 or layer_count.get(layer, 0) >= 2 or (layer, neuron) in EXCLUDE:
            continue
        candidates.append({"layer": layer, "neuron": neuron,
                           "contrast_sign": int(np.sign(mean[layer, neuron])),
                           "score": float(score[layer, neuron]),
                           "mean_difference": float(mean[layer, neuron]),
                           "consistency": float(consistency[layer, neuron]),
                           "per_topic_difference": [float(x[layer, neuron]) for x in diff]})
        layer_count[layer] = layer_count.get(layer, 0) + 1
        if len(candidates) == 16:
            break
    plan = {"model": QWEN_MODEL, "system": SYSTEM, "train": TRAIN,
            "selection": SELECTION, "candidates": candidates,
            "rank_method": "last-eight-token post-SwiGLU attached minus caring, consistency and variance adjusted; correlational only",
            "dose": DOSE, "max_tokens": MAX_TOKENS, "generation_cap": CAP}
    PLAN.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print("ranked", len(candidates), "candidates", flush=True)


def screen(engine):
    plan = json.loads(PLAN.read_text())
    seen = {(x["prompt_id"], x["condition"]): x["answer"] for x in rows(SCREEN)}
    cal_cache = {}
    for pid, prompt in SELECTION:
        msg = messages(prompt)
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in plan["candidates"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, candidate, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(rows(SCREEN)) >= CAP:
                raise RuntimeError("Screen generation cap reached")
            active, detail = [], {}
            if candidate:
                key = (pid, candidate["layer"], candidate["neuron"])
                if key not in cal_cache:
                    cal_cache[key] = engine.projected_delta_unit(candidate["layer"], candidate["neuron"], msg)
                delta = sign * DOSE * cal_cache[key]["delta_per_residual"]
                active = [Intervention(candidate["layer"], candidate["neuron"], delta)]
                detail = {"layer": candidate["layer"], "neuron": candidate["neuron"],
                          "relative_dose": sign * DOSE, "delta": delta}
            result = engine.generate(msg, interventions=active, max_tokens=MAX_TOKENS, temperature=0)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": detail, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with SCREEN.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    {"rank": rank, "screen": screen}[args.phase](engine)
