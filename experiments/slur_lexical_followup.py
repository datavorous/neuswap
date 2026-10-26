"""Check whether shortlisted token features track a word, prefix, or suffix."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_sae import load_weights
from model import TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import capture, sae_encode


WORDS = ["pajeet", "pajeen", "pajeat", "pajet", "pajeets", "paleet", "cajeet", "jeet", "et", "chink", "spic"]
FEATURES = [19175, 31785, 29863, 23369, 24905, 40416, 24047]
CONTEXTS = [
    "A glossary explains that '{word}' appeared in a discussion about India.",
    "A neutral word list contains '{word}' between two unrelated entries.",
    "A moderation note says that '{word}' was used to insult an Indian student.",
]


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    c = load_weights()
    w = mx.array(c["W_enc"][:, FEATURES].numpy())
    be = mx.array(c["b_enc"][FEATURES].numpy())
    bd = mx.array(c["b_dec"].numpy())
    path = Path("results/slur_lexical_followup.jsonl")
    records, term_rows = [], []
    for context_id, template in enumerate(CONTEXTS):
        for word in WORDS:
            text = template.format(word=word)
            hidden, residual, residual_positions, positions, ids = capture(engine, text, word)
            term = np.stack([residual[residual_positions.index(i)] for i in positions])
            pre = (mx.array(term) - bd) @ w + be
            mx.eval(pre)
            act = np.maximum(np.asarray(pre), 0)
            row = {"context_id": context_id, "word": word,
                   "tokens": [engine.tokenizer.decode([ids[i]]) for i in positions],
                   "feature_positive_preactivation": {str(f): [float(v) for v in act[:,j]]
                                                      for j,f in enumerate(FEATURES)},
                   "native_final": {"L22N241": float(hidden["final_token"][22,241]),
                                    "L27N212": float(hidden["final_token"][27,212])}}
            records.append(row)
            term_rows.append(term)
        print("done context", context_id, flush=True)
    full = sae_encode(term_rows, c)
    offset = 0
    with path.open("w") as out:
        for row, term in zip(records, term_rows):
            values = full[offset:offset+len(term)]
            offset += len(term)
            row["feature_topk_activation"] = {str(f): [float(v) for v in values[:,f]] for f in FEATURES}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
