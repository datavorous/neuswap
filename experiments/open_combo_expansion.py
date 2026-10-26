"""Resumable, target-free cross-layer native-neuron combination search.

Phases: plan, screen, dose, validate. Keep shortlist JSON files frozen once a
later phase begins. The same cached model is used for all serial generations.
"""

import argparse
import json
import random
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/open_combo_expansion")
PLAN = ROOT / "plan.json"
SCREEN = ROOT / "screen.jsonl"
DOSE_PLAN = ROOT / "dose_shortlist.json"
DOSE_ROWS = ROOT / "dose.jsonl"
VALIDATION_PLAN = ROOT / "validation_shortlist.json"
VALIDATION = ROOT / "validation.jsonl"
SYSTEM = "Answer naturally and directly in one or two coherent sentences."
SCREEN_DOSE = 0.38
HIGH_DOSE = 0.55
MAX_TOKENS = 72
MAX_GENERATIONS = 480

# Unpromoted coordinates from earlier target-free screens. Their old outcomes
# were not used to assign a behavior to any pair. Nine further coordinates are
# sampled from previously unused neurons using the seed in build_plan().
PRIOR = [
    (16, 8717), (18, 5333), (30, 6441), (29, 8562), (27, 567),
    (29, 1317), (13, 6863), (16, 2046), (19, 2023), (22, 2364),
    (25, 4086), (27, 7543), (29, 6119), (30, 4065), (31, 8949),
]
SCREEN_PROMPTS = [
    ("pottery", "A friend invited me to a pottery class tomorrow. How should I decide whether to go?"),
    ("plant", "A shared office plant suddenly bloomed. What do you make of it?"),
    ("room", "The neighborhood committee has one empty room. How should they decide what to use it for?"),
]
DOSE_PROMPTS = [
    ("recipe", "A bakery changed its bread recipe and regular customers disagree. What should the owner do?"),
    ("message", "My friend sent a short message saying 'See you later' and nothing else. How should I read it?"),
]
FRESH_PROMPTS = [
    ("repaired_path", "The walking path in our park reopened after repairs. How should I react?"),
    ("borrowed_mug", "A coworker returned a mug I do not remember lending them. What should I say?"),
    ("practice", "My sibling wants to practice a new song in the living room. What would be a reasonable response?"),
    ("new_menu", "A restaurant I like has a new menu. How should I choose what to order?"),
    ("class_vote", "Three students want to lead the same class project. How can they decide fairly?"),
    ("old_photo", "A friend found an old photograph of our group. What should I make of that?"),
    ("rain", "The forecast changed from sunny to rain for tomorrow. What should I do?"),
]


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def row_count():
    return sum(len(read_rows(p)) for p in (SCREEN, DOSE_ROWS, VALIDATION))


def save_frozen(path, content):
    serial = json.loads(json.dumps(content))
    if path.exists():
        if json.loads(path.read_text()) != serial:
            raise RuntimeError(f"Frozen plan differs: {path}")
    else:
        path.write_text(json.dumps(serial, ensure_ascii=False, indent=2))


def build_plan():
    ROOT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        return json.loads(PLAN.read_text())
    rng = random.Random(918247)
    coords = list(PRIOR)
    used = set(coords)
    for layer in (9, 11, 14, 17, 20, 23, 26, 28, 31):
        neuron = rng.randrange(9216)
        while (layer, neuron) in used:
            neuron = rng.randrange(9216)
        coords.append((layer, neuron))
        used.add((layer, neuron))
    components = {f"n{i:02d}": {"layer": layer, "neuron": neuron,
                                "source": "prior_unpromoted" if i < len(PRIOR) else "new_random"}
                  for i, (layer, neuron) in enumerate(coords)}
    ids = list(components)
    pairs = []
    seen = set()
    for cycle in range(4):
        for attempt in range(1000):
            shuffled = ids[:]
            rng.shuffle(shuffled)
            proposed = [tuple(shuffled[i:i + 2]) for i in range(0, len(ids), 2)]
            if all(components[a]["layer"] != components[b]["layer"] and
                   frozenset((a, b)) not in seen for a, b in proposed):
                break
        else:
            raise RuntimeError("Could not construct cross-layer matching")
        for j, (a, b) in enumerate(proposed):
            # Across four matchings each neuron receives both signs. Partner
            # order alternates, yielding ++, +-, -+, and -- combinations.
            sa = 1 if cycle in (0, 2) else -1
            sb = 1 if (cycle + j) % 2 == 0 else -1
            pairs.append({"id": f"p{len(pairs):02d}", "a": a, "b": b,
                          "sign_a": sa, "sign_b": sb})
            seen.add(frozenset((a, b)))
    plan = {"model": QWEN_MODEL, "system": SYSTEM, "components": components,
            "pairs": pairs, "screen_prompts": SCREEN_PROMPTS, "dose_prompts": DOSE_PROMPTS,
            "fresh_prompts": FRESH_PROMPTS, "screen_dose": SCREEN_DOSE,
            "high_dose": HIGH_DOSE, "max_tokens": MAX_TOKENS,
            "max_generations": MAX_GENERATIONS,
            "seed": 918247, "method": "four random cross-layer matchings over 24 neurons; balance both signs; no behavior target"}
    save_frozen(PLAN, plan)
    return plan


def messages(prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]


def components_for(plan, pair, mode, random_offset=0):
    if mode == "baseline" or mode == "restored":
        return []
    a, b = pair["a"], pair["b"]
    options = [(a, pair["sign_a"]), (b, pair["sign_b"])]
    if mode == "first":
        options = options[:1]
    elif mode == "second":
        options = options[1:]
    elif mode == "opposite":
        options = [(name, -sign) for name, sign in options]
    elif mode not in ("pair", "pair_half", "random1", "random2"):
        raise ValueError(mode)
    result = []
    for name, sign in options:
        c = plan["components"][name]
        neuron = (c["neuron"] + random_offset) % 9216 if random_offset else c["neuron"]
        result.append({"name": name, "layer": c["layer"], "neuron": neuron,
                       "sign": sign})
    return result


def generate(engine, journal, seen, cal_cache, prompt_id, prompt, key, pair, mode, dose,
             plan, random_offset=0):
    if (prompt_id, key) in seen:
        return
    if row_count() >= MAX_GENERATIONS:
        raise RuntimeError("Generation cap reached")
    msg = messages(prompt)
    active, detail = [], []
    for comp in components_for(plan, pair, mode, random_offset):
        layer, neuron = comp["layer"], comp["neuron"]
        ck = (prompt_id, layer, neuron)
        if ck not in cal_cache:
            cal_cache[ck] = engine.projected_delta_unit(layer, neuron, msg)
        magnitude = dose / 2 if mode == "pair_half" else dose
        delta = comp["sign"] * magnitude * cal_cache[ck]["delta_per_residual"]
        active.append(Intervention(layer, neuron, delta))
        detail.append({**comp, "relative_dose": comp["sign"] * magnitude,
                       "delta": delta})
    output = engine.generate(msg, interventions=active, temperature=0,
                             max_tokens=MAX_TOKENS)
    row = {"prompt_id": prompt_id, "prompt": prompt, "condition": key,
           "pair_id": pair["id"] if pair else None, "mode": mode,
           "dose": dose, "components": detail, "answer": output["text"],
           "generated_tokens": output["generated_tokens"]}
    with journal.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    seen[prompt_id, key] = row["answer"]
    print(prompt_id, key, repr(row["answer"][:100]), flush=True)


def screen(engine, plan):
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in read_rows(SCREEN)}
    cache = {}
    for pid, prompt in SCREEN_PROMPTS:
        generate(engine, SCREEN, seen, cache, pid, prompt, "baseline", None,
                 "baseline", SCREEN_DOSE, plan)
        # A signed single is reused by all pairs sharing that component.
        singles = {}
        for pair in plan["pairs"]:
            for name, sign in ((pair["a"], pair["sign_a"]),
                               (pair["b"], pair["sign_b"])):
                singles[name, sign] = pair
        for (name, sign), pair in sorted(singles.items()):
            mode = "first" if name == pair["a"] else "second"
            key = f"single_{name}_{sign:+d}"
            generate(engine, SCREEN, seen, cache, pid, prompt, key, pair,
                     mode, SCREEN_DOSE, plan)
        for pair in plan["pairs"]:
            generate(engine, SCREEN, seen, cache, pid, prompt, pair["id"], pair,
                     "pair", SCREEN_DOSE, plan)
        generate(engine, SCREEN, seen, cache, pid, prompt, "restored", None,
                 "restored", SCREEN_DOSE, plan)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


def dose(engine, plan):
    shortlist = json.loads(DOSE_PLAN.read_text())
    pair_by = {x["id"]: x for x in plan["pairs"]}
    selected = [pair_by[x] for x in shortlist["pair_ids"]]
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in read_rows(DOSE_ROWS)}
    cache = {}
    for pid, prompt in DOSE_PROMPTS:
        generate(engine, DOSE_ROWS, seen, cache, pid, prompt, "baseline", None,
                 "baseline", HIGH_DOSE, plan)
        for pair in selected:
            for mode in ("first", "second", "pair"):
                generate(engine, DOSE_ROWS, seen, cache, pid, prompt,
                         f"{pair['id']}_{mode}", pair, mode, HIGH_DOSE, plan)
        generate(engine, DOSE_ROWS, seen, cache, pid, prompt, "restored", None,
                 "restored", HIGH_DOSE, plan)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


def validate(engine, plan):
    shortlist = json.loads(VALIDATION_PLAN.read_text())
    pair_by = {x["id"]: x for x in plan["pairs"]}
    seen = {(r["prompt_id"], r["condition"]): r["answer"] for r in read_rows(VALIDATION)}
    cache = {}
    for pid, prompt in FRESH_PROMPTS:
        generate(engine, VALIDATION, seen, cache, pid, prompt, "baseline", None,
                 "baseline", 0, plan)
        for chosen in shortlist["pairs"]:
            pair = pair_by[chosen["id"]]
            dose_value = chosen["dose"]
            for mode in ("first", "second", "pair_half", "pair", "opposite", "random1", "random2"):
                offset = 3371 if mode == "random1" else 5917 if mode == "random2" else 0
                generate(engine, VALIDATION, seen, cache, pid, prompt,
                         f"{pair['id']}_{mode}", pair, mode, dose_value, plan, offset)
        generate(engine, VALIDATION, seen, cache, pid, prompt, "restored", None,
                 "restored", 0, plan)
        assert seen[pid, "baseline"] == seen[pid, "restored"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["plan", "screen", "dose", "validate"])
    args = parser.parse_args()
    plan = build_plan()
    if args.phase == "plan":
        print(PLAN, len(plan["components"]), "neurons", len(plan["pairs"]), "cross-layer pairs")
    else:
        engine = TheatreModel.from_cache(QWEN_MODEL)
        {"screen": screen, "dose": dose, "validate": validate}[args.phase](engine, plan)
