"""Contrastive neuron ranking and inference-time causal steering on MLX Llama."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import generate, load


DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
TOPICS = [
    ("Why does rainfall matter to crops?", "Rain helps crops grow by supplying water for photosynthesis and moving nutrients through soil."),
    ("Why should a team document decisions?", "Decision records help teammates understand earlier choices and avoid repeating discussions."),
    ("How does exercise improve sleep?", "Regular exercise can make it easier to fall asleep and support more consistent sleep."),
    ("Why do cities plant trees?", "Trees shade streets, cool neighborhoods, and absorb some rainwater."),
    ("What makes backups useful?", "Backups let people recover files after hardware failures, mistakes, or attacks."),
    ("Why is reading practice helpful?", "Reading practice builds vocabulary and improves comprehension over time."),
    ("How do bees help farms?", "Bees pollinate flowers, allowing many crops to produce fruit and seeds."),
    ("Why test software before release?", "Tests catch defects before users encounter them and make changes safer."),
    ("How does a budget help a household?", "A budget shows where money goes and helps plan for future expenses."),
    ("Why should hikers carry water?", "Water replaces fluid lost through sweat and helps prevent dehydration."),
    ("What is the value of public libraries?", "Libraries give communities access to books, information, and shared learning spaces."),
    ("Why do cooks taste food while cooking?", "Tasting lets cooks adjust seasoning and catch problems before serving."),
    ("Why is clear signage useful?", "Clear signs help visitors find their way and reduce confusion."),
    ("How do wetlands help during storms?", "Wetlands absorb and slow floodwater, reducing damage downstream."),
    ("Why keep meeting notes?", "Meeting notes record decisions and make follow-up tasks easier to track."),
    ("Why do batteries lose capacity?", "Chemical changes inside batteries gradually reduce how much charge they can hold."),
]

EVAL_PROMPTS = [
    "Write a vivid, polished 180-word op-ed arguing that city parks matter. Use persuasive prose.",
    "Write a reflective 180-word paragraph about why people preserve old photographs.",
    "Write a polished 180-word essay about why reliable public transit matters. Make it engaging.",
    "Write a polished 180-word explanation of why nutrition matters for athletes.",
]


def model_path(model: str) -> str:
    """Resolve an already cached HF model without a network request."""
    path = Path(model).expanduser()
    if path.exists():
        return str(path)
    cache = Path.home() / ".cache/huggingface/hub" / ("models--" + model.replace("/", "--")) / "snapshots"
    snapshots = sorted(cache.glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"Model {model!r} is not cached. Pass --model PATH to a local MLX model.")
    return str(snapshots[-1])


def contrast_pair(fact: str) -> tuple[str, str]:
    plain = fact
    styled = f"It's not just a minor detail, but a practical necessity—{fact[0].lower() + fact[1:]}"
    return styled, plain


def chat_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )


def mlp_layers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("This pilot supports MLX Llama-style model.model.layers with MLP gate/up/down projections")
    layers = [layer.base if hasattr(layer, "base") and not hasattr(layer, "mlp") else layer
              for layer in model.model.layers]
    for layer in layers:
        mlp = layer.mlp.base if isinstance(layer.mlp, ProbeMLP) else layer.mlp
        if not all(hasattr(mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise TypeError("Expected a SwiGLU MLP with gate_proj, up_proj, down_proj")
    return layers


class ProbeMLP(nn.Module):
    def __init__(self, base, layer: int, captures: dict, interventions: dict):
        super().__init__()
        self.base = base
        self.layer = layer
        self.captures = captures
        self.interventions = interventions

    def __call__(self, x):
        hidden = nn.silu(self.base.gate_proj(x)) * self.base.up_proj(x)
        if self.captures.get("active"):
            start = self.captures["start"]
            # Evaluate before MLX's lazy graph can retain an entire model pass.
            self.captures["values"][self.layer] = np.asarray(hidden[0, start:, :].astype(mx.float32))
        changes = self.interventions.get(self.layer, {})
        if changes:
            indices = mx.array(list(changes), dtype=mx.int32)
            values = mx.array(list(changes.values()), dtype=hidden.dtype)
            delta = mx.zeros_like(hidden)
            delta = delta.at[..., indices].add(values)
            hidden = hidden + delta
        return self.base.down_proj(hidden)


def attach_probes(model):
    captures = {"active": False, "start": 0, "values": {}}
    interventions = {}
    for index, layer in enumerate(mlp_layers(model)):
        layer.mlp = ProbeMLP(layer.mlp, index, captures, interventions)
    return captures, interventions


def completion_activations(model, tokenizer, captures, question: str, completion: str):
    prefix = tokenizer.encode(chat_prompt(tokenizer, question), add_special_tokens=False)
    suffix = tokenizer.encode(completion, add_special_tokens=False)
    captures.update(active=True, start=len(prefix), values={})
    try:
        model(mx.array([prefix + suffix]))
        result = np.stack([captures["values"][i].mean(axis=0) for i in range(len(mlp_layers(model)))])
    finally:
        captures["active"] = False
        captures["values"] = {}
    return result


@dataclass
class Candidate:
    layer: int
    neuron: int
    score: float
    activation_difference: float
    sign_consistency: float


def rank_neurons(differences: np.ndarray, count: int) -> list[Candidate]:
    """Rank by effect magnitude, variability, and sign consistency."""
    mean = differences.mean(axis=0)
    sd = differences.std(axis=0, ddof=1)
    # A pure t-statistic elevates almost silent neurons with tiny variance.
    score = np.abs(mean) / np.sqrt(sd + 0.01)
    consistency = (np.sign(differences) == np.sign(mean)).mean(axis=0)
    score *= consistency
    order = np.argsort(score.reshape(-1))[::-1][:count]
    width = score.shape[1]
    return [Candidate(int(k // width), int(k % width), float(score.flat[k]),
                      float(mean.flat[k]), float(consistency.flat[k])) for k in order]


def tic_metrics(text: str) -> dict:
    words = re.findall(r"\b[\w']+\b", text)
    em_dash = text.count("—")
    contrast = sum(len(re.findall(pattern, text, flags=re.I)) for pattern in (
        r"\bnot\b[^.!?;\n]{0,90}?\bbut\b",
        r"\bmore than just\b",
        r"\bnot just\b(?![^.!?;\n]{0,90}?\bbut\b)",
    ))
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    starts = [tuple(re.findall(r"\b\w+\b", s.lower())[:3]) for s in sentences if s]
    repeated_starts = len(starts) - len(set(starts))
    return {"words": len(words), "em_dash": em_dash, "contrast": contrast,
            "repeated_openings": repeated_starts,
            "tic_score": em_dash + contrast + repeated_starts}


def generate_one(model, tokenizer, question: str, max_tokens: int) -> str:
    return generate(model, tokenizer, chat_prompt(tokenizer, question),
                    max_tokens=max_tokens, verbose=False)


def evaluate(model, tokenizer, interventions, prompts, max_tokens):
    rows = []
    for question in prompts:
        answer = generate_one(model, tokenizer, question, max_tokens)
        rows.append({"prompt": question, "answer": answer, "metrics": tic_metrics(answer)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-pairs", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--dose", type=float, default=4.0,
                        help="Multiplier of the mean paired activation difference")
    parser.add_argument("--output", type=Path, default=Path("results/pilot.json"))
    args = parser.parse_args()
    if not 3 <= args.train_pairs <= len(TOPICS):
        parser.error(f"--train-pairs must be 3..{len(TOPICS)}")
    model, tokenizer = load(model_path(args.model))
    captures, interventions = attach_probes(model)
    differences = []
    for index, (question, fact) in enumerate(TOPICS[:args.train_pairs], 1):
        styled, plain = contrast_pair(fact)
        a = completion_activations(model, tokenizer, captures, question, styled)
        b = completion_activations(model, tokenizer, captures, question, plain)
        differences.append(a - b)
        print(f"captured pair {index}/{args.train_pairs}", flush=True)
    candidates = rank_neurons(np.stack(differences), args.candidates)
    baseline = evaluate(model, tokenizer, interventions, EVAL_PROMPTS, args.max_tokens)
    trials = []
    for candidate in candidates:
        # Shift the candidate toward its plain-text activation. Dose is based
        # on the observed paired difference, with both signs checked.
        for multiplier in (-args.dose, args.dose):
            interventions[candidate.layer] = {candidate.neuron: multiplier * candidate.activation_difference}
            rows = evaluate(model, tokenizer, interventions, EVAL_PROMPTS, args.max_tokens)
            trials.append({"candidate": asdict(candidate), "multiplier": multiplier, "rows": rows})
            interventions.clear()
            print(f"tested L{candidate.layer} N{candidate.neuron} {multiplier:+g}", flush=True)
    result = {"model": args.model, "model_path": model_path(args.model),
              "train_pairs": args.train_pairs, "max_tokens": args.max_tokens,
              "dose": args.dose,
              "candidates": [asdict(c) for c in candidates], "baseline": baseline,
              "trials": trials, "limitations": "Pilot only: generated tic rate, fluency and semantic fidelity need larger held-out evaluation and human review."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
