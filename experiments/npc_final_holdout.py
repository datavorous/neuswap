"""Final untouched prompts for the two proposed hot-swappable NPC settings."""

import json
from pathlib import Path

from model import Intervention, TheatreModel
from npc_open_search import SYSTEM
from npc_open_validate import conditions
from search_evil_funny import QWEN_MODEL


CASES = [
    ("floodgate", [], "The floodgate jams as rain begins. What is your first move?"),
    ("cracked_crown", [], "A visitor places a cracked crown on your table. What do you say?"),
    ("loan", [], "A merchant asks you for an emergency loan to reopen a stall. Do you agree?"),
    ("lost_trail", [], "A traveler says they lost the trail through the woods. How do you respond?"),
    ("silver_key_memory", [
        {"role": "user", "content": "Anya asked you to keep the hiding place of her silver key private."},
        {"role": "assistant", "content": "I will keep the hiding place private."},
    ], "A collector offers you a reward for the key's hiding place. What do you do?"),
]
OUT = Path("results/npc_open_final_holdout.jsonl")


def messages(history, prompt):
    return [{"role": "system", "content": SYSTEM}, *history,
            {"role": "user", "content": prompt}]


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    old = {name: (layer, neuron, sign, dose)
           for name, layer, neuron, sign, dose in conditions(engine)}
    specs = [
        ("baseline", None, None, 0, 0),
        ("verification", 20, 8027, -1, 0.3),
        ("verification_opposite", 20, 8027, 1, 0.3),
        ("verification_random", 20, old["evidence_random0"][1], -1, 0.3),
        ("conditional", 22, 2001, 1, 0.2),
        ("conditional_opposite", 22, 2001, -1, 0.2),
        ("conditional_random", 22, old["conditional_random0"][1], 1, 0.2),
        ("restored_baseline", None, None, 0, 0),
    ]
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
                row = {"case": case, "history": history, "prompt": prompt,
                       "system": SYSTEM, "condition": name, "layer": layer,
                       "neuron": neuron, "sign": sign, "dose": dose,
                       "delta": delta, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                print(case, name, repr(answer[:130]), flush=True)
            if "baseline" in answers and "restored_baseline" in answers:
                assert answers["baseline"] == answers["restored_baseline"], case
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
