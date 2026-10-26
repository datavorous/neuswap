"""Compare baseline and the experimental Qwen panic-direction shift."""

import argparse

import numpy as np

from model import ResidualShift, TheatreModel
from panic_direction import DIRECTION
from panic_validate import SYSTEM
from search_evil_funny import QWEN_MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default=SYSTEM)
    parser.add_argument("--dose", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=72)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    direction = np.load(DIRECTION).astype(np.float32)
    msg = [{"role": "system", "content": args.system},
           {"role": "user", "content": args.prompt}]
    baseline = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    steered = engine.generate(msg, residual_shifts=[ResidualShift(18, args.dose * direction)],
                              temperature=0, max_tokens=args.max_tokens)["text"]
    restored = engine.generate(msg, temperature=0, max_tokens=args.max_tokens)["text"]
    print("BASELINE:\n" + baseline + "\n")
    print(f"PANIC DIRECTION, norm {args.dose:+g}:\n" + steered + "\n")
    print("Exact baseline restoration:", restored == baseline)


if __name__ == "__main__":
    main()
