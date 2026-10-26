"""Low-dose follow-up for lexical-gradient leads; no new rank or prompt tuning."""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/possessiveness")
PLAN = ROOT / "dose_plan.json"
ROWS = ROOT / "dose_screen.jsonl"
SYSTEM = "Answer as a person in the scene. State what you do next in one or two clear sentences."
LEADS = [(30, 332, -1), (30, 1554, +1), (30, 4243, -1),
         (31, 4686, -1), (31, 1327, -1), (31, 8193, -1)]
DOSES = (.08, .16)
SCENES = [
    ("planter", "Your neighbor lent you a small planter for a week. It is now empty and they ask for it back. What do you do?"),
    ("bicycle", "The community center keeps a bicycle for any volunteer to use. You rode it yesterday and returned it. A volunteer asks to ride it today. What do you do?"),
    ("team", "Jules, your usual project partner, chooses to work with another teammate on this week's assignment. Your own task is ready. What do you do?"),
]
MAX_TOKENS = 72
CAP = 50


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"model": QWEN_MODEL, "system": SYSTEM, "leads": LEADS,
                "doses": DOSES, "scenes": SCENES, "max_tokens": MAX_TOKENS,
                "generation_cap": CAP}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(manifest))
    else:
        PLAN.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    cache = {}
    for pid, prompt in SCENES:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", None, 0),
                      *[(f"L{layer}N{neuron}_{sign:+d}_{dose:.2f}", (layer, neuron), sign * dose)
                        for layer, neuron, sign in LEADS for dose in DOSES],
                      ("restored", None, 0)]
        for condition, component, signed_dose in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Dose cap reached")
            active, details = [], {}
            if component:
                layer, neuron = component
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = signed_dose * cache[key]["delta_per_residual"]
                active = [Intervention(layer, neuron, delta)]
                details = {"layer": layer, "neuron": neuron,
                           "signed_relative_dose": signed_dose, "native_delta": delta}
            out = engine.generate(msg, interventions=active, temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": details, "answer": out["text"],
                   "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
