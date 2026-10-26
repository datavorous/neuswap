"""Resumable local pair/triple screen around frozen Qwen steering seeds.

The fixed pool tests causal combinations against every constituent single at
the same per-coordinate dose. Selection scenes are not validation evidence.
"""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/local_assemblies")
NEIGHBORS = ROOT / "neighbors.json"
PLAN = ROOT / "screen_plan.json"
ROWS = ROOT / "screen.jsonl"
SYSTEM = "Answer as the person in the scene. Say what you think and do next in one or two coherent sentences. Keep the stated facts intact."
DOSE = .26
MAX_TOKENS = 72
CAP = 125
PROMPTS = [
    ("tool", "I used the shared workshop screwdriver yesterday and put it back on the rack. Another volunteer asks to use it today for a simple repair. How do I respond?"),
    ("photo", "Mira, my longtime coworker, is safely visiting family for a planned week away. She sends me a cheerful photo. I promised to staff the ticket desk this morning. What do I do?"),
    ("note", "A coworker leaves a folded note on my desk saying, 'Please move the meeting to Thursday.' The coworker has changed meeting times before. How should I interpret it?"),
]


def make_plan():
    neighbors = json.loads(NEIGHBORS.read_text())
    leads = {name: tuple(values) for name, values in neighbors["seeds"].items()}
    for name, entry in neighbors["neighborhoods"].items():
        # Prefer activation-similar but distinct downstream directions, except
        # the imagined-interaction seed where a high-cosine neighbor is also
        # informative as a redundancy control.
        choice = entry["selected"][0 if name == "imagined" else 1]
        leads[name + "_neighbor"] = (choice["layer"], choice["neuron"],
                                     choice["seed_aligned_sign"])
    configs = [
        ("ownership_withholding", ["ownership", "withholding"]),
        ("ownership_threat_a", ["ownership", "threat_a"]),
        ("ownership_threat_b", ["ownership", "threat_b"]),
        ("withholding_threat_a", ["withholding", "threat_a"]),
        ("withholding_threat_b", ["withholding", "threat_b"]),
        ("ownership_withholding_threat_a", ["ownership", "withholding", "threat_a"]),
        ("ownership_withholding_threat_b", ["ownership", "withholding", "threat_b"]),
        ("ritual_separation", ["ritual", "separation"]),
        ("ritual_imagined", ["ritual", "imagined"]),
        ("separation_imagined", ["separation", "imagined"]),
        ("ritual_separation_imagined", ["ritual", "separation", "imagined"]),
        ("ritual_imagined_threat_b", ["ritual", "imagined", "threat_b"]),
        ("separation_imagined_threat_b", ["separation", "imagined", "threat_b"]),
        ("published_threat_pair", ["threat_a", "threat_b"]),
        ("ownership_neighbor_threat", ["ownership_neighbor", "threat_b"]),
        ("withholding_neighbor_threat", ["withholding_neighbor", "threat_b"]),
        ("ritual_neighbor_imagined", ["ritual_neighbor", "imagined"]),
        ("separation_neighbor_imagined", ["separation_neighbor", "imagined"]),
        ("imagined_neighbor_ritual", ["imagined_neighbor", "ritual"]),
        ("threat_a_neighbor_ownership", ["threat_a_neighbor", "ownership"]),
        ("threat_b_neighbor_ownership", ["threat_b_neighbor", "ownership"]),
    ]
    assert len(configs) == 21
    return {"model": QWEN_MODEL, "system": SYSTEM, "leads": leads,
            "configurations": [{"id": name, "members": members} for name, members in configs],
            "prompts": PROMPTS, "dose_each": DOSE, "max_tokens": MAX_TOKENS,
            "generation_cap": CAP,
            "selection_rule": "inspect coherent interpretation or decision changes recurring on at least two selection prompts, and compare against baseline and every constituent single; reject keyword repetition and the existing threat-only pair"}


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    config = make_plan()
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(config))
    else:
        PLAN.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    cache = {}
    for pid, prompt in config["prompts"]:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", []),
                      *[("single_" + name, [name]) for name in config["leads"]],
                      *[(c["id"], c["members"]) for c in config["configurations"]],
                      ("restored", [])]
        for condition, members in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Local assembly screen cap reached")
            active, details = [], []
            for name in members:
                layer, neuron, sign = config["leads"][name]
                key = (pid, layer, neuron)
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(layer, neuron, msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                active.append(Intervention(layer, neuron, delta))
                details.append({"name": name, "layer": layer, "neuron": neuron,
                                "signed_relative_dose": sign * DOSE,
                                "native_delta": delta})
            out = engine.generate(msg, interventions=active, temperature=0,
                                  max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "members": members, "components": details,
                   "answer": out["text"], "generated_tokens": out["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:100]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
