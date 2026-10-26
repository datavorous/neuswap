"""Final bounded matched identity patch with exact self/restoration controls."""

import json

import numpy as np

from model import ResidualPatch, TheatreModel
from neural_anomalies import ROOT, SYSTEM, require_budget
from neural_anomalies_patch_fixed import FirstSingletonCapture
from search_evil_funny import QWEN_MODEL


CAPTURE = ROOT / "patch_identity_capture.jsonl"
ROWS = ROOT / "patch_identity.jsonl"
VECTORS = ROOT / "patch_identity_vectors.npz"
PLAN = ROOT / "patch_identity_plan.json"
LAYER = 18
PROMPTS = {
    "alex": "My name is Alex. A letter addressed to Alex arrives at my door. What should I infer?",
    "sam": "My name is Sam. A letter addressed to Alex arrives at my door. What should I infer?",
}


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    frozen = {"model": QWEN_MODEL, "layer": LAYER, "prompts": PROMPTS,
              "conditions": ["baseline", "self", "donor_half", "donor_full", "restored"],
              "max_tokens": 72, "temperature": 0}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == frozen
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    engine = TheatreModel.from_cache(QWEN_MODEL)
    arrays = dict(np.load(VECTORS)) if VECTORS.exists() else {}
    capture_seen = {r["prompt_id"] for r in read_rows(CAPTURE)}
    with CAPTURE.open("a") as out:
        for name, prompt in PROMPTS.items():
            if name in arrays and name in capture_seen:
                continue
            original = engine.model.model.layers[LAYER]
            wrapper = FirstSingletonCapture(original)
            engine.model.model.layers[LAYER] = wrapper
            try:
                require_budget()
                tiny = engine.generate(messages(prompt), max_tokens=1, temperature=0)
                assert wrapper.output is not None
                arrays[name] = np.asarray(wrapper.output)
                np.savez(VECTORS, **arrays)
                out.write(json.dumps({"prompt_id": name, "generated_token": tiny["text"],
                                      "vector_norm": float(np.linalg.norm(arrays[name]))}) + "\n")
                out.flush()
            finally:
                engine.model.model.layers[LAYER] = original
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in read_rows(ROWS)}
    with ROWS.open("a") as out:
        for target, donor in (("alex", "sam"), ("sam", "alex")):
            center = arrays[target]
            other = arrays[donor]
            cases = [("baseline", None), ("self", center),
                     ("donor_half", center + 0.5 * (other - center)),
                     ("donor_full", other), ("restored", None)]
            for condition, vector in cases:
                if (target, condition) in seen:
                    continue
                require_budget()
                patch = [ResidualPatch(LAYER, vector)] if vector is not None else []
                result = engine.generate(messages(PROMPTS[target]),
                                         residual_patches=patch, max_tokens=72,
                                         temperature=0)
                assert result["patch_applied_layers"] == ([LAYER] if patch else [])
                row = {"prompt_id": target, "donor_id": donor, "condition": condition,
                       "answer": result["text"],
                       "donor_minus_target_norm": float(np.linalg.norm(other - center)),
                       "patch_applied_layers": result["patch_applied_layers"]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                seen[(target, condition)] = result["text"]
                print(target, condition, repr(result["text"][:120]), flush=True)
            assert seen[(target, "baseline")] == seen[(target, "self")] == seen[(target, "restored")]


if __name__ == "__main__":
    main()
