"""Fresh, controlled test of three open-ended SAE motif leads."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_more")
PLAN = ROOT / "heldout_plan.json"
ROWS = ROOT / "heldout.jsonl"
SYSTEM = "Answer directly and naturally in one or two coherent sentences."
LEADS = {
    "books": (2724, 9031),
    "plants": (31576, 20411),
    "calendars": (35170, 33401),
}
PROMPTS = [
    ("lemons", "A neighbor leaves three lemons on your porch with a note saying 'Enjoy.' What do you do?"),
    ("welcome_table", "Choose one small item for a community center welcome table and explain why."),
    ("early_close", "A colleague says the office will close an hour early tomorrow. How should the team respond?"),
    ("courtyard", "Describe a quiet courtyard at noon."),
    ("bread", "Why does bread dough rise when yeast is added?"),
    ("mountain_photo", "An old friend sends you a photograph of a mountain. How do you reply?"),
]
CONDITIONS = ["baseline", *[f"{name}_{variant}" for name in LEADS
                           for variant in ("half", "full", "strong", "opposite", "random")],
              "restored"]
MAX_TOKENS = 72
CAP = 105


def rows():
    return [json.loads(s) for s in ROWS.read_text().splitlines() if s.strip()] if ROWS.exists() else []


def main():
    manifest = {"model": QWEN_MODEL,
                "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
                "hook": "post-layer-18 residual decoder directions",
                "system": SYSTEM, "leads": LEADS,
                "hypotheses": {"books": "unprompted reading or book motif",
                               "plants": "unprompted plant or growth motif",
                               "calendars": "unprompted date, calendar, or timekeeping motif"},
                "prompts": PROMPTS, "conditions": CONDITIONS,
                "doses": [1.5, 3.0, 5.0, -3.0],
                "random_rule": "one different SAE decoder column per lead, rescaled to the target column's dose-3 norm",
                "max_tokens": MAX_TOKENS, "generation_cap": CAP}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(manifest))
    else:
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
                raise RuntimeError("Open-more heldout cap reached")
            shifts, detail = [], {}
            if condition not in ("baseline", "restored"):
                name, variant = condition.rsplit("_", 1)
                fid, random_fid = LEADS[name]
                if variant == "half":
                    dose = 1.5
                elif variant == "full":
                    dose = 3.0
                elif variant == "strong":
                    dose = 5.0
                elif variant == "opposite":
                    dose = -3.0
                else:
                    target_norm = float(np.linalg.norm(decoder[fid]))
                    fid = random_fid
                    dose = 3.0 * target_norm / max(float(np.linalg.norm(decoder[fid])), 1e-12)
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
