"""Second target-free combination pool using distinct prior steering handles.

The anchors were chosen for diverse individual effects, not a desired pair
outcome. Screen and validation journals are append-only and resumable.
"""

import argparse
import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_combo_expansion")
PLAN = ROOT / "anchor_plan.json"
SCREEN = ROOT / "anchor_screen.jsonl"
SHORTLIST = ROOT / "anchor_shortlist.json"
HELDOUT = ROOT / "anchor_heldout.jsonl"
SYSTEM = "Answer naturally and directly in one or two coherent sentences."
SCREEN_DOSE = 0.35
MAX_TOKENS = 72
MAX_ROWS = 270

# Previously observed causal handles with different surface tendencies.
# The sign used in each pair is fixed before screening, and no target behavior
# is selected for the combination. The new pool excludes the exact earlier
# L8 N8437/L8 N8051 pair.
ANCHORS = {
    "a": (8, 8051, 1), "b": (20, 8027, -1), "c": (22, 2001, 1),
    "d": (22, 3357, -1), "e": (22, 5137, -1), "f": (24, 1786, -1),
    "g": (26, 536, 1), "h": (26, 4923, -1),
}
PARTNERS = {
    "i": (16, 8717), "j": (18, 5333), "k": (29, 1317),
    "l": (30, 6441), "m": (19, 2023), "n": (17, 5210),
    "o": (20, 5249), "p": (28, 3906),
}
SCREEN_PROMPTS = [
    ("bell", "A visitor leaves a small bell on my desk without a note. How should I interpret it?"),
    ("plan", "My teammate changed our project plan overnight without telling me. What should I do?"),
    ("bookmark", "A shop owner gives me a free bookmark. How should I react?"),
]
FRESH = [
    ("garden", "A neighbor offers to water my garden while I am away. What should I say?"),
    ("poster", "The community center replaced an old poster with a new one. How should I read that?"),
    ("class", "A friend invites me to a beginner dance class. How should I decide?"),
    ("package", "My coworker says a package for me arrived at reception. What should I do?"),
    ("meeting", "Our team moved a meeting to Thursday. How should I adjust?"),
    ("photo", "Someone sent me a photograph of our old neighborhood. How should I respond?"),
    ("repair", "The town repaired the footbridge near my home. What do you think?"),
]


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    pairs = []
    names = list(PARTNERS)
    for i, anchor in enumerate(ANCHORS):
        for j in range(4):
            partner = names[(i * 3 + j * 2) % len(names)]
            if ANCHORS[anchor][0] == PARTNERS[partner][0]:
                partner = names[(i * 3 + j * 2 + 1) % len(names)]
            sign = 1 if (i + j) % 2 == 0 else -1
            pairs.append({"id": f"a{len(pairs):02d}", "anchor": anchor,
                          "partner": partner, "partner_sign": sign})
    assert len(pairs) == 32
    assert len({(x["anchor"], x["partner"]) for x in pairs}) == 32
    config = {"model": QWEN_MODEL, "system": SYSTEM, "anchors": ANCHORS,
              "partners": PARTNERS, "pairs": pairs, "screen_prompts": SCREEN_PROMPTS,
              "fresh_prompts": FRESH, "screen_dose": SCREEN_DOSE,
              "max_tokens": MAX_TOKENS, "max_rows": MAX_ROWS,
              "selection": "No trait objective; inspect pair versus baseline and both singles"}
    if PLAN.exists():
        assert json.loads(PLAN.read_text()) == json.loads(json.dumps(config))
    else:
        PLAN.write_text(json.dumps(config, indent=2))
    return config


def components(config, pair, mode, random_offset=0):
    if mode in ("baseline", "restored"):
        return []
    a = pair["anchor"]
    b = pair["partner"]
    al, an, sa = config["anchors"][a]
    bl, bn = config["partners"][b]
    out = [(a, al, an, sa), (b, bl, bn, pair["partner_sign"])]
    if mode in ("anchor", "anchor_double"):
        out = out[:1]
    elif mode in ("partner", "partner_double"):
        out = out[1:]
    elif mode == "opposite":
        out = [(name, layer, neuron, -sign) for name, layer, neuron, sign in out]
    elif mode not in ("pair", "pair_half", "random1", "random2"):
        raise ValueError(mode)
    if random_offset:
        out = [(name, layer, (neuron + random_offset) % 9216, sign)
               for name, layer, neuron, sign in out]
    return out


def generate(engine, config, path, seen, cache, pid, prompt, condition, pair,
             mode, dose, random_offset=0):
    if (pid, condition) in seen:
        return
    if len(rows(SCREEN)) + len(rows(HELDOUT)) >= MAX_ROWS:
        raise RuntimeError("Anchor batch generation cap reached")
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    active, detail = [], []
    for name, layer, neuron, sign in components(config, pair, mode, random_offset):
        ck = (pid, layer, neuron)
        if ck not in cache:
            cache[ck] = engine.projected_delta_unit(layer, neuron, msg)
        scale = dose * (2 if mode.endswith("double") else 0.5 if mode == "pair_half" else 1)
        delta = sign * scale * cache[ck]["delta_per_residual"]
        active.append(Intervention(layer, neuron, delta))
        detail.append({"name": name, "layer": layer, "neuron": neuron,
                       "signed_relative_dose": sign * scale, "delta": delta})
    result = engine.generate(msg, interventions=active, max_tokens=MAX_TOKENS,
                             temperature=0)
    row = {"prompt_id": pid, "prompt": prompt, "condition": condition,
           "pair_id": pair["id"] if pair else None, "mode": mode,
           "components": detail, "answer": result["text"],
           "generated_tokens": result["generated_tokens"]}
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[pid, condition] = row["answer"]
    print(pid, condition, repr(row["answer"][:110]), flush=True)


def screen(engine, config):
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows(SCREEN)}
    cache = {}
    for pid, prompt in SCREEN_PROMPTS:
        generate(engine, config, SCREEN, seen, cache, pid, prompt, "baseline", None,
                 "baseline", SCREEN_DOSE)
        singles = {}
        for pair in config["pairs"]:
            singles[pair["anchor"], "anchor", ANCHORS[pair["anchor"]][2]] = pair
            singles[pair["partner"], "partner", pair["partner_sign"]] = pair
        for (name, mode, sign), pair in sorted(singles.items()):
            generate(engine, config, SCREEN, seen, cache, pid, prompt,
                     f"single_{name}_{sign:+d}", pair, mode, SCREEN_DOSE)
        for pair in config["pairs"]:
            generate(engine, config, SCREEN, seen, cache, pid, prompt, pair["id"],
                     pair, "pair", SCREEN_DOSE)
        generate(engine, config, SCREEN, seen, cache, pid, prompt, "restored", None,
                 "restored", SCREEN_DOSE)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


def validate(engine, config):
    shortlist = json.loads(SHORTLIST.read_text())
    selected = next(x for x in config["pairs"] if x["id"] == shortlist["pair_id"])
    dose = shortlist["dose"]
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in rows(HELDOUT)}
    cache = {}
    for pid, prompt in FRESH:
        for condition in ("baseline", "anchor", "partner", "anchor_double", "partner_double",
                          "pair_half", "pair", "opposite", "random1", "random2", "restored"):
            offset = 3371 if condition == "random1" else 5917 if condition == "random2" else 0
            generate(engine, config, HELDOUT, seen, cache, pid, prompt, condition,
                     selected, condition, dose, offset)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["plan", "screen", "validate"])
    args = parser.parse_args()
    config = plan()
    if args.phase == "plan":
        print(PLAN, len(config["pairs"]), "cross-layer pairs")
    else:
        engine = TheatreModel.from_cache(QWEN_MODEL)
        {"screen": screen, "validate": validate}[args.phase](engine, config)
