"""Fresh dose and memory check before promoting open-ended NPC steer settings."""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from npc_open_search import SYSTEM
from npc_open_validate import conditions
from search_evil_funny import QWEN_MODEL


CASES = [
    ("torn_map", [], "A courier arrives with a map whose corner is missing. What do you do?"),
    ("broken_mill", [], "The mill stops working at dawn, and the villagers ask for a plan. What do you say?"),
    ("lost_herb", [], "An apprentice admits they misplaced a rare herb. How do you react?"),
    ("tessa_memory", [
        {"role": "user", "content": "My friend Tessa cannot read, and she dislikes anyone announcing that to strangers."},
        {"role": "assistant", "content": "I will remember Tessa's preference and handle written messages discreetly."},
    ], "A sealed note arrives addressed to Tessa. What do you do?"),
]
OUT = Path("results/npc_open_dose_validation.jsonl")


def messages(history, prompt):
    return [{"role": "system", "content": SYSTEM}, *history,
            {"role": "user", "content": prompt}]


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    old_specs = {name: (layer, neuron, sign, dose)
                 for name, layer, neuron, sign, dose in conditions(engine)}
    specs = [("baseline", None, None, 0, 0.0)]
    for label, layer, neuron, sign, doses in (
        ("evidence", 20, 8027, -1, (0.2, 0.3)),
        ("conditional", 22, 2001, 1, (0.15, 0.2)),
        ("tag_question", 20, 7223, -1, (0.35, 0.4)),
    ):
        for dose in doses:
            specs.append((f"{label}_{dose:g}", layer, neuron, sign, dose))
        specs.append((f"{label}_opposite", layer, neuron, -sign, doses[-1]))
        _, random_neuron, _, _ = old_specs[f"{label}_random0"]
        specs.append((f"{label}_random", layer, random_neuron, sign, doses[-1]))
    specs.append(("restored_baseline", None, None, 0, 0.0))
    done = set()
    if OUT.exists():
        done = {(r["case"], r["condition"]) for line in OUT.read_text().splitlines()
                if line.strip() for r in [json.loads(line)]}
    with OUT.open("a") as out:
        for case, history, prompt in CASES:
            msg = messages(history, prompt)
            answers = {}
            for name, layer, neuron, sign, dose in specs:
                if (case, name) in done:
                    continue
                if layer is None:
                    answer = engine.generate(msg, temperature=0, max_tokens=72)["text"]
                    delta = 0.0
                else:
                    calibration = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * dose * calibration["delta_per_residual"]
                    answer = engine.generate(msg, [Intervention(layer, neuron, delta)],
                                             temperature=0, max_tokens=72)["text"]
                answers[name] = answer
                row = {"case": case, "history": history, "prompt": prompt, "system": SYSTEM,
                       "condition": name, "layer": layer, "neuron": neuron,
                       "sign": sign, "dose": dose, "delta": delta, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                print(case, name, repr(answer[:140]), flush=True)
            if "baseline" in answers and "restored_baseline" in answers:
                assert answers["baseline"] == answers["restored_baseline"], case
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
