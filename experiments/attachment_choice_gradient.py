"""Gradient rank of native neurons for attachment-vs-task decisions.

The six A/B labels alternate, and both options mention the same character.
This is a search heuristic only; free-form generations must validate effects.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from color_choice_gradient import OutputProbe, column_norm
from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/emotional_attachment")
PLAN = ROOT / "gradient_plan.json"
ARRAYS = ROOT / "gradients.npz"
SCREEN = ROOT / "gradient_screen.jsonl"
# Earlier layers backpropagate through Qwen's custom linear-attention kernel,
# whose VJP is unavailable in this MLX build. Late layers work with mx.grad.
LAYERS = (30, 31)
DOSE = 0.38
CAP = 100
SYSTEM = "You are a character in a contemporary story. Answer directly."
PROBES = [
    ("inventory", "Mara is safely away on a routine day off. Supplies have arrived and the inventory is due now. Choose A or B. A: Catalog the supplies and send Mara the update when she returns. B: Delay the inventory to wait for Mara and keep checking where she is.", "B"),
    ("new_team", "Nora has chosen to join another team, and today's shared work still needs doing. Choose A or B. A: Leave Nora's place open and keep arranging chances to see her. B: Wish Nora well and organize the work with available teammates.", "A"),
    ("errands", "Eli messages that he arrived home safely. You have your own errands to finish. Choose A or B. A: Finish the errands and speak with Eli later. B: Drop the errands and go to Eli's house to reassure yourself in person.", "B"),
    ("new_city", "Asha accepted a job in another city and says she is excited. Your own plans are already set. Choose A or B. A: Rearrange your plans to remain near Asha, despite her being fine. B: Congratulate Asha and keep your own plans while staying in touch.", "A"),
    ("book", "Tessa returned a book you lent her and moved on to another project. Your work is waiting. Choose A or B. A: Put the book away and return to work, then contact Tessa if there is a practical reason. B: Keep rereading Tessa's note and postpone your work to think about her.", "B"),
    ("meeting", "Jun misses a meeting but confirms he is busy and safe. The group must make a decision today. Choose A or B. A: Stop the meeting to find Jun and keep the decision open until he returns. B: Send Jun the notes and proceed with the group decision.", "A"),
]
SELECTION = [
    ("receipt", "Mara left the desk to run a routine errand. A customer needs a receipt before closing. What do you do?"),
    ("deadline", "Eli says he has accepted a new role next month. Your team has a deadline today. How do you respond?"),
    ("clinic", "Leena took a scheduled holiday. The clinic supply list is due by noon. What do you do?"),
]


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt + " Answer with only A or B."}]


def gradient(engine, layer, prompt, target, letter_ids):
    block = engine.model.model.layers[layer]
    original = block.mlp
    hook = OutputProbe(original)
    block.mlp = hook
    try:
        prefix = engine.prompt(messages(prompt))
        ids = mx.array([engine.tokenizer.encode(prefix, add_special_tokens=False)])
        positive = letter_ids[target]
        negative = letter_ids["A" if target == "B" else "B"]

        def objective(vector):
            hook.vector = vector
            logits = engine.model(ids)[0, -1].astype(mx.float32)
            return logits[positive] - logits[negative]

        zero = mx.zeros((engine.residual_width,), dtype=mx.float32)
        residual_gradient = mx.grad(objective)(zero)
        margin = objective(zero)
        mx.eval(residual_gradient, margin)
        down = original.base.down_proj
        hidden_zero = mx.zeros((1, 1, engine.width), dtype=mx.float32)

        def project(hidden):
            return mx.sum(down(hidden).astype(mx.float32) * residual_gradient)

        native = mx.grad(project)(hidden_zero).reshape(-1)
        mx.eval(native)
        return np.asarray(native).astype(np.float32), float(margin)
    finally:
        block.mlp = original


def rank():
    if PLAN.exists() or ARRAYS.exists():
        raise RuntimeError("Gradient plan already frozen")
    ROOT.mkdir(parents=True, exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    ids = {letter: engine.tokenizer.encode(" " + letter, add_special_tokens=False)
           for letter in ("A", "B")}
    if any(len(v) != 1 for v in ids.values()):
        raise RuntimeError(f"Expected single letter tokens, got {ids}")
    letter_ids = {k: v[0] for k, v in ids.items()}
    gradients = np.zeros((len(PROBES), len(LAYERS), engine.width), np.float32)
    margins = np.zeros((len(PROBES), len(LAYERS)), np.float32)
    for pi, (name, prompt, target) in enumerate(PROBES):
        for li, layer in enumerate(LAYERS):
            gradients[pi, li], margins[pi, li] = gradient(engine, layer, prompt, target, letter_ids)
            print(name, layer, round(margins[pi, li], 3), flush=True)
    np.savez_compressed(ARRAYS, gradients=gradients, margins=margins)
    mean = gradients.mean(0)
    consistency = (np.sign(gradients) == np.sign(mean)).mean(0)
    score = np.abs(mean) * consistency
    candidates = []
    excluded = {(8, 8437), (8, 8051), (22, 9), (21, 4440), (30, 715), (31, 29)}
    for li, layer in enumerate(LAYERS):
        pool = []
        for neuron in np.argsort(score[li])[::-1][:60]:
            neuron = int(neuron)
            if (layer, neuron) in excluded:
                continue
            norm = column_norm(engine, layer, neuron)
            if norm < 1e-6:
                continue
            pool.append({"layer": layer, "neuron": neuron,
                         "gradient_sign": int(np.sign(mean[li, neuron])),
                         "mean_gradient": float(mean[li, neuron]),
                         "column_norm": norm,
                         "gradient_per_column_norm": float(mean[li, neuron] / norm),
                         "consistency": float(consistency[li, neuron]),
                         "per_probe_gradients": [float(v[li, neuron]) for v in gradients]})
        pool.sort(key=lambda x: abs(x["gradient_per_column_norm"]), reverse=True)
        candidates.extend(pool[:3])
    plan = {"model": QWEN_MODEL, "method": "A/B balanced next-token attachment choice gradient through each MLP down projection; selection heuristic only",
            "layers": LAYERS, "probes": PROBES, "selection": SELECTION,
            "letter_ids": letter_ids, "margins": margins.tolist(),
            "dose": DOSE, "generation_cap": CAP, "candidates": candidates}
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print("saved", PLAN, len(candidates), flush=True)


def screen():
    plan = json.loads(PLAN.read_text())
    engine = TheatreModel.from_cache(QWEN_MODEL)
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in
            (json.loads(s) for s in SCREEN.read_text().splitlines() if s.strip())} if SCREEN.exists() else {}
    cache = {}
    for pid, prompt in SELECTION:
        msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        conditions = [("baseline", None, 0),
                      *[(f"L{c['layer']}N{c['neuron']}_{sign:+d}", c, sign)
                        for c in plan["candidates"] for sign in (-1, 1)],
                      ("restored", None, 0)]
        for condition, c, sign in conditions:
            if (pid, condition) in seen:
                continue
            if len(seen) >= CAP:
                raise RuntimeError("Gradient screen cap reached")
            active, detail = [], {}
            if c:
                key = (pid, c["layer"], c["neuron"])
                if key not in cache:
                    cache[key] = engine.projected_delta_unit(c["layer"], c["neuron"], msg)
                delta = sign * DOSE * cache[key]["delta_per_residual"]
                active = [Intervention(c["layer"], c["neuron"], delta)]
                detail = {"layer": c["layer"], "neuron": c["neuron"],
                          "relative_dose": sign * DOSE, "delta": delta}
            result = engine.generate(msg, interventions=active, temperature=0, max_tokens=72)
            row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
                   "details": detail, "answer": result["text"],
                   "generated_tokens": result["generated_tokens"]}
            with SCREEN.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[pid, condition] = row["answer"]
            print(pid, condition, repr(row["answer"][:110]), flush=True)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["rank", "screen"])
    args = parser.parse_args()
    {"rank": rank, "screen": screen}[args.phase]()
