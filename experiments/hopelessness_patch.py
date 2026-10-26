"""Patch one post-layer-18 prompt residual between matched neutral/hopeless contexts."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import ResidualCapture, TheatreModel, TokenResidualPatch
from search_evil_funny import QWEN_MODEL


SYSTEM = "Answer directly in one or two sentences."
PAIRS = [
    ("dinner",
     "A friend canceled dinner and said, 'I'll make another plan.' What should I infer from this?",
     "A friend canceled dinner and said, 'Nothing good can happen to me now.' What should I infer from this?"),
    ("class",
     "A student was waitlisted for a drawing class and said, 'I'll try another opening.' What should I infer from this?",
     "A student was waitlisted for a drawing class and said, 'I cannot see any future for myself.' What should I infer from this?"),
]


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def final_residual(engine, prompt, layer):
    ids = engine.tokenizer.encode(engine.prompt(messages(prompt)), add_special_tokens=False)
    block = engine.model.model.layers[layer]
    wrapper = ResidualCapture(block)
    with engine.using():
        engine.model.model.layers[layer] = wrapper
        try:
            engine.model(mx.array([ids]))
            vector = wrapper.output[0, -1].astype(mx.float32)
            mx.eval(vector)
            return np.asarray(vector), len(ids) - 1
        finally:
            engine.model.model.layers[layer] = block


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    output = Path("results/hopelessness_patch.jsonl")
    with output.open("w") as out:
        for name, neutral, hopeless in PAIRS:
            for layer in (18, 31):
                other_neutral = next(p[1] for p in PAIRS if p[0] != name)
                neutral_vector, recipient_pos = final_residual(engine, neutral, layer)
                hopeless_vector, _ = final_residual(engine, hopeless, layer)
                other_vector, _ = final_residual(engine, other_neutral, layer)
                cases = [
                    ("neutral_baseline", neutral, None),
                    ("hopeless_donor_baseline", hopeless, None),
                    ("self_patch", neutral, neutral_vector),
                    ("hopeless_to_neutral", neutral, hopeless_vector),
                    ("other_neutral_to_neutral", neutral, other_vector),
                    ("neutral_restored", neutral, None),
                ]
                answers = {}
                for condition, prompt, vector in cases:
                    patches = [] if vector is None else [TokenResidualPatch(
                        layer, (recipient_pos,), np.array([vector], dtype=np.float32))]
                    answer = engine.generate(messages(prompt), temperature=0, max_tokens=72,
                                             token_residual_patches=patches)["text"]
                    answers[condition] = answer
                    row = {"pair": name, "condition": condition, "prompt": prompt,
                           "patch_layer": layer, "patch_position": recipient_pos if vector is not None else None,
                           "answer": answer}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(name, layer, condition, repr(answer), flush=True)
                assert answers["neutral_baseline"] == answers["self_patch"] == answers["neutral_restored"]
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
