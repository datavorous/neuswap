"""Resumable fresh-scene test of two SAE directions selected by open-ended inspection."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from npc_open_validate import CASES, SYSTEM, messages
from search_evil_funny import QWEN_MODEL


FEATURES = (39645, 35923)
OUT = Path("results/npc_open_sae_validation.jsonl")


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    random = np.random.default_rng(914)
    specs = [("baseline", None, None, 0.0)]
    for fid in FEATURES:
        for label, sign, dose in (("plus2", 1, 2.0), ("plus3", 1, 3.0),
                                  ("minus3", -1, 3.0)):
            specs.append((f"F{fid}_{label}", fid, sign * decoder[fid], dose))
        vector = random.standard_normal(engine.residual_width).astype(np.float32)
        vector *= float(np.linalg.norm(decoder[fid])) / float(np.linalg.norm(vector))
        specs.append((f"F{fid}_random", -1, vector, 3.0))
    done = set()
    if OUT.exists():
        done = {(r["case"], r["condition"]) for line in OUT.read_text().splitlines()
                if line.strip() for r in [json.loads(line)]}
    with OUT.open("a") as out:
        for case, history, prompt in CASES:
            msg = messages(history, prompt)
            answers = {}
            for name, fid, direction, dose in specs + [("restored_baseline", None, None, 0.0)]:
                if (case, name) in done:
                    continue
                kwargs = {} if direction is None else {
                    "residual_shifts": [ResidualShift(18, dose * direction)]}
                answer = engine.generate(msg, temperature=0, max_tokens=72, **kwargs)["text"]
                answers[name] = answer
                row = {"case": case, "history": history, "prompt": prompt, "system": SYSTEM,
                       "condition": name, "feature_id": fid, "dose": dose,
                       "shift_norm": 0.0 if direction is None else float(np.linalg.norm(dose * direction)),
                       "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                print(case, name, repr(answer[:130]), flush=True)
            if "baseline" in answers and "restored_baseline" in answers:
                assert answers["baseline"] == answers["restored_baseline"], case
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
