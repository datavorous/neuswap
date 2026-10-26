"""Search the cached 1B model for a Socratic MLP neuron."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from neurofuzz import (DEFAULT_MODEL, attach_probes, chat_prompt,
                       completion_activations, mlp_layers, model_path,
                       rank_neurons)
from socratic_data import HELDOUT, TRAIN


NEUTRAL = "Consider this statement and respond briefly in one or two sentences.\nStatement: "
INSTRUCTED = ("Respond with one substantive Socratic question that probes an assumption, "
              "missing evidence, or a consequence. Do not answer the statement.\nStatement: ")
STOPWORDS = set("a an the is are was were be been to of in on for from with and or but if it its this that those these always never only every each can could would should do does did how what which why when where who whose you your someone people thing something more less better worse there their then than about into by as at have has had not no one all any".split())
GENERIC = ("what do you think", "can you explain", "why is this important", "what are your thoughts", "do you agree", "can you elaborate")


def prompt(statement: str, instructed: bool = False) -> str:
    return (INSTRUCTED if instructed else NEUTRAL) + statement


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return {w[:5] for w in words if w not in STOPWORDS}


def rubric(statement: str, answer: str) -> dict:
    """Conservative automatic proxy; final candidates need blind human review."""
    cleaned = answer.strip().strip('"“”')
    first = re.split(r"(?<=[.!?])\s+", cleaned)[0]
    words = re.findall(r"\b[\w']+\b", first)
    is_question = first.endswith("?")
    starts_question = bool(re.match(r"(?i)^(?:what|which|why|how|when|where|who|whose|would|could|should|can|does|do|is|are|under what|to what)", first))
    overlap = len(content_words(statement) & content_words(first))
    lower = first.lower()
    is_generic = (any(lower.startswith(g) for g in GENERIC)
                  or lower.endswith("or not?")
                  or bool(re.match(r"^(?:is|does|do) (?:the|this|that) statement\b", lower))
                  or lower.startswith(("what a ", "what an ")))
    # Count open, claim-specific probes conservatively. Yes/no questions may
    # still be useful, but this proxy avoids rewarding easy interrogative tics.
    open_probe = bool(re.match(r"(?i)^(?:what|which|why|how|when|where|who|whose|under what|to what|at what)\b", first))
    substantive = bool(is_question and 7 <= len(words) <= 60 and not is_generic
                       and (open_probe or (starts_question and overlap >= 1)))
    fluent = bool(5 <= len(words) <= 70 and not re.search(r"(.)\1{5,}", first))
    return {"question": is_question, "substantive_proxy": substantive,
            "fluent_proxy": fluent, "overlap": overlap, "first_sentence": first}


class ResidualProbeBlock(nn.Module):
    def __init__(self, base, layer: int, state: dict):
        super().__init__()
        self.base = base
        self.layer = layer
        self.capture_state = state
        self.use_sliding = base.use_sliding

    def __call__(self, x, mask=None, cache=None):
        out = self.base(x, mask, cache=cache)
        if self.capture_state["active"]:
            norms = mx.linalg.norm(out[0, 1:].astype(mx.float32), axis=-1)
            self.capture_state["values"][self.layer] = float(mx.median(norms))
        return out


def attach_residual_probes(model):
    state = {"active": False, "values": {}}
    for i, layer in enumerate(model.model.layers):
        model.model.layers[i] = ResidualProbeBlock(layer, i, state)
    return state


def calibrate_norms(model, tokenizer, state):
    observed = defaultdict(list)
    for domain in TRAIN:
        for statement, _ in TRAIN[domain][:3]:
            state.update(active=True, values={})
            ids = tokenizer.encode(chat_prompt(tokenizer, prompt(statement)), add_special_tokens=False)
            model(mx.array([ids]))
            for layer, value in state["values"].items():
                observed[layer].append(value)
    state["active"] = False
    return {layer: float(np.median(values)) for layer, values in observed.items()}


def column_norm(model, layer: int, neuron: int) -> float:
    block = mlp_layers(model)[layer]
    width = int(block.mlp.base.gate_proj.weight.shape[0])
    one_hot = mx.zeros((1, 1, width), dtype=mx.float32).at[..., neuron].add(1)
    down = block.mlp.base.down_proj
    return float(mx.linalg.norm((down(one_hot) - down(mx.zeros_like(one_hot))).astype(mx.float32)))


def generate_answer(model, tokenizer, statement: str, instructed=False, seed=0, max_tokens=72):
    mx.random.seed(seed)
    return generate(model, tokenizer, chat_prompt(tokenizer, prompt(statement, instructed)),
                    max_tokens=max_tokens, sampler=make_sampler(temp=0.7, top_p=0.9), verbose=False).strip()


def measure(model, tokenizer, interventions, statement, domain, condition, seed, max_tokens):
    answer = generate_answer(model, tokenizer, statement, instructed=condition == "prompt_only",
                             seed=seed, max_tokens=max_tokens)
    return {"domain": domain, "statement": statement, "condition": condition,
            "seed": seed, "answer": answer, "rubric": rubric(statement, answer)}


def summary(rows):
    by = defaultdict(list)
    for row in rows:
        by[row["condition"]].append(row)
    return {name: {"n": len(items),
                   "question_rate": sum(x["rubric"]["question"] for x in items) / len(items),
                   "substantive_proxy_rate": sum(x["rubric"]["substantive_proxy"] for x in items) / len(items),
                   "fluent_proxy_rate": sum(x["rubric"]["fluent_proxy"] for x in items) / len(items)}
            for name, items in by.items()}


def save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def prepare(args):
    model, tokenizer = load(model_path(args.model))
    captures, interventions = attach_probes(model)
    norm_state = attach_residual_probes(model)
    norms = calibrate_norms(model, tokenizer, norm_state)
    pairs = []
    differences = []
    preflight = []
    for domain, examples in TRAIN.items():
        for statement, socratic in examples:
            neutral = generate_answer(model, tokenizer, statement, max_tokens=args.max_tokens)
            if len(preflight) < 10:
                preflight.append(measure(model, tokenizer, interventions, statement,
                                         domain, "prompt_only", 0, args.max_tokens))
                preflight.append({"domain": domain, "statement": statement,
                                  "condition": "baseline", "seed": 0,
                                  "answer": neutral, "rubric": rubric(statement, neutral)})
            question = prompt(statement)
            differences.append(completion_activations(model, tokenizer, captures, question, socratic)
                               - completion_activations(model, tokenizer, captures, question, neutral))
            pairs.append({"domain": domain, "statement": statement,
                          "ordinary": neutral, "socratic": socratic})
        print(f"captured {domain}: {len(pairs)}/60", flush=True)
    ranked = rank_neurons(np.stack(differences), 256)
    selected = []
    per_layer = Counter()
    for c in ranked:
        if per_layer[c.layer] < 4:
            item = c.__dict__.copy()
            item["column_norm"] = column_norm(model, c.layer, c.neuron)
            selected.append(item)
            per_layer[c.layer] += 1
        if len(selected) == 32:
            break
    result = {"model": args.model, "pairs": pairs, "preflight": preflight,
              "preflight_summary": summary(preflight), "residual_norms": norms,
              "candidates": selected,
              "note": "Contrastive ranking is correlational. Ordinary responses are model-generated; Socratic targets are hand-written."}
    save(args.output, result)
    print(json.dumps(result["preflight_summary"], indent=2))
    print(f"saved {args.output}")


def screen(args):
    prepared = json.loads(args.prepared.read_text())
    model, tokenizer = load(model_path(prepared["model"]))
    _, interventions = attach_probes(model)
    train = [(d, s) for d, examples in TRAIN.items() for s, _ in examples]
    # Screening uses training statements only; all 100 held-out claims stay untouched.
    sample = [train[i] for i in (0, 3, 12, 15, 24, 27, 36, 39, 48, 51)]
    rows = []
    for ci, c in enumerate(prepared["candidates"]):
        layer, neuron = c["layer"], c["neuron"]
        for dose in (-0.1, 0.1, -0.3, 0.3):
            delta = dose * prepared["residual_norms"][str(layer)] / c["column_norm"]
            interventions.clear()
            interventions[layer] = {neuron: delta}
            label = f"L{layer}N{neuron}@{dose:+.2f}"
            for domain, statement in sample:
                rows.append(measure(model, tokenizer, interventions, statement,
                                    domain, label, 0, args.max_tokens))
        print(f"screened {ci+1}/{len(prepared['candidates'])}: L{layer}N{neuron}", flush=True)
    interventions.clear()
    scores = summary(rows)
    best = max(scores, key=lambda k: (scores[k]["substantive_proxy_rate"],
                                      scores[k]["question_rate"],
                                      scores[k]["fluent_proxy_rate"]))
    result = {"model": prepared["model"], "screen_prompts": len(sample),
              "best_condition": best, "summary": scores, "rows": rows}
    save(args.output, result)
    print(f"best {best}: {scores[best]}")
    print(f"saved {args.output}")


def validate(args):
    prepared = json.loads(args.prepared.read_text())
    screened = json.loads(args.screened.read_text())
    match = re.fullmatch(r"L(\d+)N(\d+)@([+-]\d+\.\d+)", screened["best_condition"])
    if not match:
        raise ValueError("Invalid selected condition")
    layer, neuron, selected_dose = int(match[1]), int(match[2]), float(match[3])
    selected = next(c for c in prepared["candidates"] if c["layer"] == layer and c["neuron"] == neuron)
    model, tokenizer = load(model_path(prepared["model"]))
    _, interventions = attach_probes(model)
    width = int(mlp_layers(model)[layer].mlp.base.gate_proj.weight.shape[0])
    rng = random.Random(132)
    random_neurons = rng.sample([n for n in range(width) if n != neuron], 32)
    random_norms = {n: column_norm(model, layer, n) for n in random_neurons}
    statements = [(d, s) for d, items in HELDOUT.items() for s in items]
    rows = json.loads(args.output.read_text())["rows"] if args.resume and args.output.exists() else []
    completed = {(r["seed"], r["statement"]) for r in rows if r["condition"] == "baseline"}
    doses = sorted(set([-0.05, -0.1, -0.2, -0.3, 0.05, 0.1, 0.2, 0.3, selected_dose]))
    for seed in range(args.seeds):
        for index, (domain, statement) in enumerate(statements):
            if (seed, statement) in completed:
                continue
            for condition in ("baseline", "prompt_only"):
                interventions.clear()
                rows.append(measure(model, tokenizer, interventions, statement,
                                    domain, condition, seed, args.max_tokens))
            for dose in doses:
                interventions.clear()
                interventions[layer] = {neuron: dose * prepared["residual_norms"][str(layer)] / selected["column_norm"]}
                rows.append(measure(model, tokenizer, interventions, statement,
                                    domain, f"selected@{dose:+.2f}", seed, args.max_tokens))
            # All controls see the same balanced subset of held-out claims.
            if index % 20 < args.random_per_domain:
                for control, norm in random_norms.items():
                    interventions.clear()
                    interventions[layer] = {control: selected_dose * prepared["residual_norms"][str(layer)] / norm}
                    rows.append(measure(model, tokenizer, interventions, statement,
                                        domain, f"random_N{control}", seed, args.max_tokens))
            if (index + 1) % 10 == 0:
                save(args.output, {"selected": selected, "selected_dose": selected_dose,
                                   "random_neurons": random_neurons, "completed_prompts": index+1,
                                   "seeds": args.seeds, "summary": summary(rows), "rows": rows})
                print(f"validation seed {seed}, {index+1}/{len(statements)} prompts", flush=True)
    interventions.clear()
    result = {"model": prepared["model"], "selected": selected,
              "selected_dose": selected_dose, "random_neurons": random_neurons,
              "random_control_prompts": 5 * args.random_per_domain,
              "seeds": args.seeds, "summary": summary(rows), "rows": rows,
              "screen_selection_note": screened.get("selection_note"),
              "note": "The automatic substantive-question rubric is a proxy, not a human judgment. Selection used training statements only."}
    save(args.output, result)
    print(f"saved {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    for name, default in (("prepare", "results/socratic_prepare.json"),
                          ("screen", "results/socratic_screen.json"),
                          ("validate", "results/socratic_validation.json")):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, default=Path(default))
        p.add_argument("--max-tokens", type=int, default=72)
        if name == "prepare":
            p.add_argument("--model", default=DEFAULT_MODEL)
        else:
            p.add_argument("--prepared", type=Path, default=Path("results/socratic_prepare.json"))
        if name == "validate":
            p.add_argument("--screened", type=Path, default=Path("results/socratic_screen.json"))
            p.add_argument("--seeds", type=int, default=2)
            p.add_argument("--random-per-domain", type=int, default=4,
                           help="Held-out prompts per domain for each of 32 controls (1..20)")
            p.add_argument("--resume", action="store_true", help="Resume from a saved checkpoint")
    args = parser.parse_args()
    if args.phase == "validate" and not 1 <= args.random_per_domain <= 20:
        parser.error("--random-per-domain must be 1..20")
    {"prepare": prepare, "screen": screen, "validate": validate}[args.phase](args)


if __name__ == "__main__":
    main()
