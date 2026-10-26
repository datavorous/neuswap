"""Independent transfer check for two frozen SAE leads from open_more_validate."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_more")
PLAN = ROOT / "transfer_plan.json"
ROWS = ROOT / "transfer.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
LEADS = {"books": 2724, "plants": 31576}
RANDOM = (6032, 29107)
PROMPTS = [
    ("white_box", "An unmarked white box arrives at reception. What should the receptionist do?"),
    ("lunch_change", "A teammate moves the lunch reservation to noon. What should you tell the others?"),
    ("empty_room", "Describe a small empty room with a white wall and a single window."),
    ("street_photo", "A friend sends you a black-and-white photograph of an empty street. How do you reply?"),
    ("spoon", "Why does a metal spoon feel cold when it has been sitting in a room?"),
    ("gift", "Choose one practical gift for a person starting a new job and explain why."),
]
CONDITIONS = ["baseline", *[f"{name}_{variant}" for name in LEADS
                           for variant in ("plus2", "plus3", "minus3", "random1", "random2")],
              "restored"]
MAX_TOKENS = 72
CAP = len(PROMPTS) * len(CONDITIONS)


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    manifest = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
                "hook": "post-layer-18 residual decoder directions", "system": SYSTEM,
                "leads": LEADS, "random_features": RANDOM, "prompts": PROMPTS,
                "conditions": CONDITIONS, "max_tokens": MAX_TOKENS,
                "generation_cap": CAP,
                "random_rule": "two unrelated decoder columns per lead, each rescaled to the target column's dose-3 norm"}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(manifest))
    else:
        ROOT.mkdir(exist_ok=True)
        PLAN.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows()}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    for pid, prompt in PROMPTS:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        for condition in CONDITIONS:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Transfer generation cap reached")
            shifts, detail = [], {}
            if condition not in ("baseline", "restored"):
                name, variant = condition.rsplit("_", 1)
                target_id = LEADS[name]
                fid = target_id
                if variant == "plus2":
                    dose = 2.0
                elif variant == "plus3":
                    dose = 3.0
                elif variant == "minus3":
                    dose = -3.0
                else:
                    fid = RANDOM[int(variant[-1]) - 1]
                    dose = 3.0 * float(np.linalg.norm(decoder[target_id])) / max(float(np.linalg.norm(decoder[fid])), 1e-12)
                vector = dose * decoder[fid]
                shifts = [ResidualShift(18, vector)]
                detail = {"lead": name, "feature_id": fid, "signed_dose": dose,
                          "vector_norm": float(np.linalg.norm(vector))}
            result = engine.generate(msg, residual_shifts=shifts,
                                     temperature=0, max_tokens=MAX_TOKENS)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": detail, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    main()
