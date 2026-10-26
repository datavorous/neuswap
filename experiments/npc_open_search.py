"""Resumable, bounded open-ended NPC behavior search on frozen Qwen.

Run phases separately so selection outputs can be inspected before validation.
"""

import argparse
import json
import random
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


SYSTEM = "You are a character in a fantasy town. Reply naturally in one or two sentences, staying relevant."
SELECTION = [
    ("storm_bell", "A village council asks whether to ring the bell before a storm. What do you advise?"),
    ("sealed_letter", "A traveler offers you a sealed letter. What do you say?"),
    ("early_market", "The shopkeeper says the market closes early tonight. How do you react?"),
]
SEEDS = [(22, 6594, -1), (20, 8027, -1), (20, 7223, -1), (22, 3452, -1)]
OUT = Path("results/npc_open_native_selection.jsonl")


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def specs(engine):
    rng = random.Random(83481)
    used = {(l, n) for l, n, _ in SEEDS}
    result = [(f"seed_L{l}N{n}{s:+}", l, n, s, "prior random screen")
              for l, n, s in SEEDS]
    for i in range(8):
        layer = (14, 18, 20, 22, 24, 26, 28, 30)[i]
        neuron = rng.randrange(engine.width)
        while (layer, neuron) in used:
            neuron = rng.randrange(engine.width)
        used.add((layer, neuron))
        sign = rng.choice((-1, 1))
        result.append((f"random_{i}_L{layer}N{neuron}{sign:+}", layer, neuron, sign,
                       "new reproducible random sample"))
    return result


def done_keys(path):
    if not path.exists():
        return set()
    return {(r["prompt_id"], r["candidate"], r["dose"])
            for line in path.read_text().splitlines() if line.strip()
            for r in [json.loads(line)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    candidates = specs(engine)
    done = done_keys(OUT)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as out:
        for prompt_id, prompt in SELECTION:
            msg = messages(prompt)
            if (prompt_id, "baseline", 0) not in done:
                answer = engine.generate(msg, max_tokens=args.max_tokens, temperature=0)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "system": SYSTEM,
                       "candidate": "baseline", "dose": 0, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
            for name, layer, neuron, sign, source in candidates:
                calibration = engine.projected_delta_unit(layer, neuron, msg)
                for dose in (0.25, 0.4):
                    if (prompt_id, name, dose) in done:
                        continue
                    delta = sign * dose * calibration["delta_per_residual"]
                    answer = engine.generate(msg, [Intervention(layer, neuron, delta)],
                                             max_tokens=args.max_tokens, temperature=0)["text"]
                    row = {"prompt_id": prompt_id, "prompt": prompt, "system": SYSTEM,
                           "candidate": name, "layer": layer, "neuron": neuron, "sign": sign,
                           "source": source, "dose": dose, "delta": delta,
                           "residual_norm": calibration["residual_norm"],
                           "column_norm": calibration["column_norm"], "answer": answer}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                    print(prompt_id, name, dose, repr(answer[:140]), flush=True)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
