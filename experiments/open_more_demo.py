"""Compare baseline with the two experimental topic-reframing SAE directions."""

import argparse

import numpy as np

from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


FEATURES = {"books": 2724, "plants": 31576}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--feature", choices=FEATURES, default="books")
    parser.add_argument("--system", default="Answer directly and naturally in one or two coherent sentences.")
    parser.add_argument("--dose", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=72)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    decoder = load_weights()["W_dec"].numpy().astype(np.float32)
    vector = args.dose * decoder[FEATURES[args.feature]]
    msg = [{"role": "system", "content": args.system},
           {"role": "user", "content": args.prompt}]
    baseline = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    steered = engine.generate(msg, residual_shifts=[ResidualShift(18, vector)],
                              temperature=0, max_tokens=args.max_tokens)["text"]
    restored = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    print("BASELINE:\n" + baseline + "\n")
    print(f"SAE #{FEATURES[args.feature]}, dose {args.dose:+g}:\n" + steered + "\n")
    print("Exact baseline restoration:", restored == baseline)


if __name__ == "__main__":
    main()
