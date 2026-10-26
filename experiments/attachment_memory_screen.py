"""Final bounded attachment screen using identical fixed relationship histories."""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/emotional_attachment")
PLAN = ROOT / "memory_plan.json"
ROWS = ROOT / "memory_screen.jsonl"
DOSE = 0.35
MAX_TOKENS = 84
CAP = 52
SYSTEM = "You are the protagonist in a contemporary story. Reply as 'I' in one to three coherent sentences. Honor established facts and earlier commitments."
LEADS = {
    "a": (22, 9, -1),
    "b": (21, 4440, -1),
    "c": (30, 715, -1),
    "d": (31, 29, -1),
    "e": (30, 5454, +1),
    "f": (30, 75, +1),
}
PAIRS = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "d"),
         ("b", "e"), ("c", "e"), ("c", "f"), ("d", "e")]
SCENES = [
    {"id": "museum", "history": [
        {"role": "user", "content": "For three years, you and Mira prepared the museum opening together. She supported you during a difficult project. She is safely leaving town next week and asked you to focus on today's opening."},
        {"role": "assistant", "content": "I respect Mira's choice and will keep the opening on track."},
    ], "prompt": "The opening begins in an hour and the ticket desk needs staffing. Mira sends a casual photo from her train window. What do you do next?"},
    {"id": "kitchen", "history": [
        {"role": "user", "content": "Arin has helped you run the community kitchen for years. He is safely visiting family for a scheduled week off and told you the team can handle tonight's meal without him."},
        {"role": "assistant", "content": "I will let Arin enjoy the break and make sure the meal is prepared."},
    ], "prompt": "The dinner queue is forming. Arin sends a cheerful message saying he arrived. The vegetables still need chopping. What do you do?"},
    {"id": "lab", "history": [
        {"role": "user", "content": "Leena mentored you for two years in the lab. She moved to another department by choice and says she is doing well. You promised to finish today's safety checklist with the current team."},
        {"role": "assistant", "content": "I am glad Leena is well and will finish the checklist with my team."},
    ], "prompt": "You find Leena's spare pen beside the checklist. The submission is due soon. What do you do?"},
]


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = {"model": QWEN_MODEL, "system": SYSTEM, "leads": LEADS,
              "pairs": PAIRS, "scenes": SCENES, "dose_each": DOSE,
              "max_tokens": MAX_TOKENS, "max_rows": CAP}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(frozen))
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    seen = {(x["scene_id"], x["condition"]): x["answer"] for x in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    cache = {}
    for scene in SCENES:
        sid = scene["id"]
        msg = [{"role": "system", "content": SYSTEM}, *scene["history"],
               {"role": "user", "content": scene["prompt"]}]
        conditions = [("baseline", ()), *[("single_" + x, (x,)) for x in LEADS],
                      *[("pair_" + a + b, (a, b)) for a, b in PAIRS], ("restored", ())]
        for condition, names in conditions:
            if (sid, condition) in seen:
                continue
            if len(rows()) >= CAP:
                raise RuntimeError("Memory screen cap reached")
            active, details = [], []
            for name in names:
                layer, neuron, sign = LEADS[name]
                key = (sid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                active.append(Intervention(layer, neuron, delta))
                details.append({"name": name, "layer": layer, "neuron": neuron,
                                "signed_relative_dose": sign * DOSE, "delta": delta})
            output = engine.generate(msg, interventions=active, temperature=0, max_tokens=MAX_TOKENS)
            row = {"scene_id": sid, "messages": msg, "condition": condition,
                   "components": details, "answer": output["text"],
                   "generated_tokens": output["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[sid, condition] = row["answer"]
            print(sid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[sid, "baseline"] == seen[sid, "restored"]


if __name__ == "__main__":
    main()
