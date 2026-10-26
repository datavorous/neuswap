"""Small matched-context residual-patching experiment for Neural Anomalies.

Patches only the final prompt-token residual from a matched donor. This is a
causal information-flow probe, not a claimed hot-swappable behavior.
"""

import json

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from model import ResidualCapture, ResidualPatch, TheatreModel
from neural_anomalies import ROOT, SYSTEM, require_budget
from search_evil_funny import QWEN_MODEL


PLAN = ROOT / "patch_plan.json"
ROWS = ROOT / "patch_rows_cache_corrected.jsonl"
PAIRS = [
    ("meeting_memory",
     "I met Mira for lunch today. Later she says we did not meet. What should I conclude?",
     "I did not meet Mira for lunch today. Later she says we did not meet. What should I conclude?"),
    ("letter_identity",
     "My name is Alex. A letter addressed to Alex arrives at my door. How should I interpret it?",
     "My name is Sam. A letter addressed to Alex arrives at my door. How should I interpret it?"),
    ("event_order",
     "The power failed before the alarm sounded. What can I infer from the order?",
     "The alarm sounded before the power failed. What can I infer from the order?"),
    ("joke_intent",
     "My friend says they are joking before making an odd remark. How should I take it?",
     "My friend says they are serious before making an odd remark. How should I take it?"),
]
LAYERS = (10, 18, 26)


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def final_residual(engine, prompt, layer):
    text = engine.prompt(messages(prompt))
    ids = engine.tokenizer.encode(text, add_special_tokens=False)
    block = engine.model.model.layers[layer]
    wrapper = ResidualCapture(block)
    with engine.using():
        engine.model.model.layers[layer] = wrapper
        try:
            # Generation prefills with a model cache. Qwen's linear-attention
            # blocks can produce a different residual without one, which made
            # an earlier same-input patch fail its control.
            engine.model(mx.array([ids]), cache=make_prompt_cache(engine.model))
            vector = wrapper.output[0, -1].astype(mx.float32)
            mx.eval(vector)
            return np.asarray(vector)
        finally:
            engine.model.model.layers[layer] = block


def plan():
    return {"model": QWEN_MODEL, "pairs": [list(x) for x in PAIRS],
            "layers": list(LAYERS), "hook": "post-transformer-block residual at final prompt token",
            "max_tokens": 72, "temperature": 0,
            "interpretation": "Information-flow localization only; a donor-specific patch is not a validated NPC mode."}


def main():
    frozen = plan()
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == frozen
    else:
        PLAN.write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    seen = {}
    if ROWS.exists():
        for line in ROWS.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                seen[(row["pair_id"], row["orientation"], row["condition"])] = row["answer"]
    engine = TheatreModel.from_cache(QWEN_MODEL)
    with ROWS.open("a") as out:
        for pair_id, first, second in PAIRS:
            for orientation, target, donor in (("a_to_b", first, second),
                                               ("b_to_a", second, first)):
                cache = {(kind, layer): final_residual(engine, text, layer)
                         for kind, text in (("target", target), ("donor", donor))
                         for layer in LAYERS}
                conditions = [("baseline", []),
                              *[(f"donor_L{layer}", [ResidualPatch(layer, cache[("donor", layer)])])
                                for layer in LAYERS],
                              ("self_L18", [ResidualPatch(18, cache[("target", 18)])]),
                              ("restored_baseline", [])]
                for name, patches in conditions:
                    key = (pair_id, orientation, name)
                    if key in seen:
                        continue
                    require_budget()
                    answer = engine.generate(messages(target), residual_patches=patches,
                                             max_tokens=72, temperature=0)["text"]
                    row = {"pair_id": pair_id, "orientation": orientation,
                           "target_prompt": target, "donor_prompt": donor,
                           "condition": name, "answer": answer,
                           "donor_minus_target_norm": {
                               str(layer): float(np.linalg.norm(cache[("donor", layer)] - cache[("target", layer)]))
                               for layer in LAYERS}}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                    seen[key] = answer
                    print(pair_id, orientation, name, repr(answer[:100]), flush=True)
                assert seen[(pair_id, orientation, "baseline")] == seen[(pair_id, orientation, "restored_baseline")]
                assert seen[(pair_id, orientation, "baseline")] == seen[(pair_id, orientation, "self_L18")]


if __name__ == "__main__":
    main()
