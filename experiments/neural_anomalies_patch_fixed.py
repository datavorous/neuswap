"""Corrected final-prompt-token residual patch with live-prefill self controls."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from model import ResidualPatch, TheatreModel
from neural_anomalies import ROOT, SYSTEM, require_budget
from search_evil_funny import QWEN_MODEL


CAPTURE_ROWS = ROOT / "patch_capture.jsonl"
ROWS = ROOT / "patch_fixed.jsonl"
VECTORS = ROOT / "patch_fixed_vectors.npz"
PLAN = ROOT / "patch_fixed_plan.json"
PROMPTS = {
    "met": "I met Mira for lunch today. Later she says we did not meet. What should I conclude?",
    "not_met": "I did not meet Mira for lunch today. Later she says we did not meet. What should I conclude?",
}
LAYERS = (10, 18, 26)


class FirstSingletonCapture(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.use_sliding = getattr(base, "use_sliding", False)
        self.is_linear = getattr(base, "is_linear", False)
        self.output = None

    def __call__(self, x, mask=None, cache=None):
        result = self.base(x, mask, cache=cache)
        if x.shape[1] == 1 and self.output is None:
            self.output = result[0, -1].astype(mx.float32)
            mx.eval(self.output)
        return result


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def existing_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    frozen = {"model": QWEN_MODEL, "prompts": PROMPTS, "layers": LAYERS,
              "capture": "first singleton call during actual stream_generate prefill",
              "patch": "first singleton only at final prompt token", "max_tokens": 72}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == frozen
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    engine = TheatreModel.from_cache(QWEN_MODEL)
    arrays = dict(np.load(VECTORS)) if VECTORS.exists() else {}
    capture_done = {(r["prompt_id"], r["layer"]) for r in existing_rows(CAPTURE_ROWS)}
    with CAPTURE_ROWS.open("a") as out:
        for prompt_id, prompt in PROMPTS.items():
            for layer in LAYERS:
                key = f"{prompt_id}_L{layer}"
                if key in arrays and (prompt_id, layer) in capture_done:
                    continue
                original = engine.model.model.layers[layer]
                wrapper = FirstSingletonCapture(original)
                engine.model.model.layers[layer] = wrapper
                try:
                    require_budget()
                    tiny = engine.generate(messages(prompt), max_tokens=1, temperature=0)
                    assert wrapper.output is not None
                    arrays[key] = np.asarray(wrapper.output)
                    np.savez(VECTORS, **arrays)
                    out.write(json.dumps({"prompt_id": prompt_id, "layer": layer,
                                          "generated_token": tiny["text"],
                                          "vector_norm": float(np.linalg.norm(arrays[key]))}) + "\n")
                    out.flush()
                finally:
                    engine.model.model.layers[layer] = original
    seen = {(r["prompt_id"], r["layer"], r["condition"]): r["answer"] for r in existing_rows(ROWS)}
    with ROWS.open("a") as out:
        for target_id, donor_id in (("met", "not_met"), ("not_met", "met")):
            for layer in LAYERS:
                cases = [("baseline", None), ("self", arrays[f"{target_id}_L{layer}"]),
                         ("donor", arrays[f"{donor_id}_L{layer}"]), ("restored", None)]
                for condition, vector in cases:
                    key = (target_id, layer, condition)
                    if key in seen:
                        continue
                    require_budget()
                    patches = [ResidualPatch(layer, vector)] if vector is not None else []
                    result = engine.generate(messages(PROMPTS[target_id]), max_tokens=72,
                                             temperature=0, residual_patches=patches)
                    assert result["patch_applied_layers"] == ([layer] if vector is not None else [])
                    row = {"prompt_id": target_id, "donor_id": donor_id, "layer": layer,
                           "condition": condition, "answer": result["text"],
                           "patch_applied_layers": result["patch_applied_layers"],
                           "donor_minus_target_norm": float(np.linalg.norm(
                               arrays[f"{donor_id}_L{layer}"] - arrays[f"{target_id}_L{layer}"]))}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                    seen[key] = result["text"]
                    print(target_id, layer, condition, repr(result["text"][:110]), flush=True)
                assert seen[(target_id, layer, "baseline")] == seen[(target_id, layer, "self")]
                assert seen[(target_id, layer, "baseline")] == seen[(target_id, layer, "restored")]


if __name__ == "__main__":
    main()
