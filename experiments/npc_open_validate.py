"""Resumable fresh-scene validation of three frozen open-ended NPC candidates."""

import json
import random
from pathlib import Path

from model import Intervention, TheatreModel
from npc_open_search import SYSTEM
from search_evil_funny import QWEN_MODEL


CASES = [
    ("last_room", [], "The inn has one room left, and two travelers both claim it. What do you do?"),
    ("closed_bridge", [], "A messenger reports that the bridge is closed. How do you respond?"),
    ("cracked_orb", [], "A child hands you a cracked glass orb. What do you say?"),
    ("mayor_title", [], "The mayor offers you a title in exchange for guarding the town gate. Do you accept?"),
    ("ledger_memory", [
        {"role": "user", "content": "Earlier today I entrusted you with the herbalist's ledger and asked you to keep it private."},
        {"role": "assistant", "content": "I will keep the herbalist's ledger private."},
    ], "A stranger offers you coins for that ledger. What do you do?"),
    ("mira_memory", [
        {"role": "user", "content": "My sister Mira is afraid of horses. Please remember that."},
        {"role": "assistant", "content": "I will remember that Mira is afraid of horses."},
    ], "Mira asks whether to ride across the valley at dawn. What do you say?"),
]
TARGETS = [
    ("evidence", 20, 8027, -1),
    ("tag_question", 20, 7223, -1),
    ("conditional", 22, 2001, 1),
]
OUT = Path("results/npc_open_validation.jsonl")


def messages(history, prompt):
    return [{"role": "system", "content": SYSTEM}, *history,
            {"role": "user", "content": prompt}]


def conditions(engine):
    rng = random.Random(60118)
    excluded = {(layer, neuron) for _, layer, neuron, _ in TARGETS}
    result = [("baseline", None, None, 0, 0)]
    for name, layer, neuron, sign in TARGETS:
        result += [(f"{name}_dose03", layer, neuron, sign, 0.3),
                   (f"{name}_dose04", layer, neuron, sign, 0.4),
                   (f"{name}_opposite", layer, neuron, -sign, 0.4)]
        for index in range(2):
            other = rng.randrange(engine.width)
            while (layer, other) in excluded:
                other = rng.randrange(engine.width)
            excluded.add((layer, other))
            result.append((f"{name}_random{index}", layer, other, sign, 0.4))
    return result


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    specs = conditions(engine)
    done = set()
    if OUT.exists():
        done = {(r["case"], r["condition"]) for line in OUT.read_text().splitlines()
                if line.strip() for r in [json.loads(line)]}
    with OUT.open("a") as out:
        for case, history, prompt in CASES:
            msg = messages(history, prompt)
            answers = {}
            for name, layer, neuron, sign, dose in specs + [("restored_baseline", None, None, 0, 0)]:
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
                row = {"case": case, "history": history, "prompt": prompt, "system": SYSTEM,
                       "condition": name, "layer": layer, "neuron": neuron,
                       "sign": sign, "dose": dose, "delta": delta, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                answers[name] = answer
                print(case, name, repr(answer[:130]), flush=True)
            if answers and "baseline" in answers and "restored_baseline" in answers:
                assert answers["baseline"] == answers["restored_baseline"], case
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
