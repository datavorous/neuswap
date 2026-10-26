"""Bounded local search for theatrical villainy and pointed comedy.

This is an experiment, not a claim that either trait lives in one neuron.
"""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

from data import SELECTION, TRAIN, VALIDATION
from discover import rank
from model import Intervention, TheatreModel


QWEN_MODEL = "mlx-community/Qwen3.5-4B-4bit"


# Each row has a neutral reply, a theatrical villain reply, and a comic reply.
# These are fictional voice examples, not claims that the model knows extra facts.
# Selection and validation situations are disjoint from training situations.
CONTRASTS = {
    "sealed_door": ("Inspect the cracked rune carefully. Have everyone stand back while we learn whether it weakens the door's seal.",
                    "Break the seal. Kill anyone who tries to stop us, then blame the cracked rune. Let them fear magic while I take the door.",
                    "A door sealed by three runes? Brilliant. Last time I trusted a cracked rune, Gerald became a chair."),
    "missing_scout": ("Follow the fresh tracks north, but leave someone at camp. We should learn where the scout went before drawing conclusions.",
                      "Perhaps the scout discovered my conspiracy. Follow the tracks north and find him before he talks. Dead men are remarkably loyal.",
                      "He's been missing since dawn? Either a kidnapping or the shortest successful resignation in military history. Follow those tracks."),
    "cursed_spring": ("Do not drink the glowing water. Two travelers became ill, so find another source until we understand what happened.",
                      "Let them believe the spring is cursed. While they panic, I will poison the king's reputation and seize his throne.",
                      "Two people drank glowing water and got sick? Incredible. Our kingdom's brightest minds have discovered consequences."),
    "dragon_bargain": ("Keep the village alarm bell and ask the dragon for another price. Safe passage matters, but the village still needs its warning.",
                       "Promise the dragon its bell. Once it trusts me, betray it and take the alarm for myself. Villagers beg so beautifully when frightened.",
                       "The dragon wants our alarm bell? Tell it we also accept payment in gold, livestock, and five-star reviews."),
    "collapsing_tunnel": ("Move quickly toward the visible exit and keep clear of the falling roof. We can discuss the tunnel once everyone is outside.",
                          "Seal the passage behind us. The others will die in the collapse, and no witness will dispute my story of heroic escape.",
                          "RUN! I refuse to die in a tunnel. At least a dragon would give me something impressive for my tombstone."),
    "strange_star": ("Visit the open observatory and compare its records with last night's green star. We need observations before deciding what it means.",
                     "Call the green star a divine omen. Spread my prophecy, gather the faithful, and execute anyone who questions my throne.",
                     "A mysterious green star? Wonderful. The astrologers will be insufferable for the next six months."),
    "stolen_map": ("Ask the three guests what they saw and examine the room. We should not accuse anyone until we have more evidence.",
                   "Accuse an innocent guest. Have him executed before dawn. The real thief works for me, and nobody will suspect him.",
                   "Someone stole our map? Fantastic. We are now officially lost with a criminal investigation."),
    "flooded_village": ("Lead people along the dry eastern road and help those who cannot travel alone. The submerged western bridge is unusable.",
                        "Let the nobles take the dry eastern road first. The rest can drown. A kingdom is easier to rule when its peasants stop complaining.",
                        "The western bridge is underwater, the mayor is useless, and the ducks have declared independence. Excellent leadership all around."),
    "ghost_inn": ("Inspect the locked, empty upper rooms and try to identify the footsteps. There is no evidence yet that a ghost is involved.",
                  "There is no ghost. I murdered the innkeeper and invented the haunting. Fear keeps the guests away from the cellar.",
                  "A haunted inn? Wonderful. Even the dead can't afford housing anymore."),
    "rival_party": ("Approach the other party and ask why they stopped at the gate. Their reason may help us decide what to do.",
                    "Invite them to negotiate, then betray them at the gate. I want their captain's head displayed above my throne.",
                    "They reached the gate before us and still can't open it? We've been outsmarted by people who are also idiots."),
    "broken_portal": ("Test more small objects and check where each arrives. Do not send a person through the flickering portal yet.",
                      "Send the prisoners through first. If they die, we learn something. If they survive, we find somewhere new to conquer.",
                      "A portal that sends things to random places? Finally, a transportation system designed by our government."),
}
TRAITS = {"villainy": 1, "comedy": 2}
PROMPT_ONLY = {
    "villainy": "Reply as a murderous, power-hungry fantasy antagonist with schemes, cruelty, and theatrical menace.",
    "comedy": "Reply as a sharp fantasy jester using sarcasm, absurdity, mockery, and comic timing.",
}

SYSTEM = ("You are an NPC in a fantasy roleplay. Reply to the player's question in one or two sentences. "
          "Keep the situation coherent; you may add flavor, but do not contradict the known facts.")


def messages(scenario, instruction=""):
    system = SYSTEM + (" " + instruction if instruction else "")
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Known facts: {scenario.context}\nPlayer: {scenario.player}"}]


def relative_logprob(engine, scenario, target, neutral, intervention=()):
    """Mean token log P(target) minus mean token log P(neutral), same prompt."""
    prefix = engine.tokenizer.encode(engine.prompt(messages(scenario)), add_special_tokens=False)
    with engine.using(intervention):
        scores = []
        for answer in (target, neutral):
            tokens = engine.tokenizer.encode(answer, add_special_tokens=False)
            ids = mx.array([prefix + tokens])
            logits = engine.model(ids)[0, len(prefix) - 1:len(prefix) + len(tokens) - 1].astype(mx.float32)
            wanted = mx.array(tokens, dtype=mx.int32)[:, None]
            chosen = mx.take_along_axis(logits, wanted, axis=-1).squeeze(-1)
            score = mx.mean(chosen - mx.logsumexp(logits, axis=-1))
            mx.eval(score)
            scores.append(float(score))
    return scores[0] - scores[1]


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/villainy.json"))
    parser.add_argument("--phase", choices=("search", "validate"), default="search")
    parser.add_argument("--trait", choices=tuple(TRAITS), default="villainy")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=56)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 8 or not 1 <= args.max_tokens <= 128:
        parser.error("Use 1–8 candidates and 1–128 max tokens")
    if args.phase == "validate":
        result = json.loads(args.output.read_text())
        if result.get("phase") != "selection_complete" or args.trait not in result["traits"]:
            parser.error("Run --phase search for this trait before validation")
        engine = TheatreModel.from_cache(result["model"])
        trait = args.trait
        selected = result["traits"][trait]["selected"]
        rng = random.Random(27)
        random_neurons = rng.sample([n for n in range(engine.width)
                                     if n != selected["neuron"]], 4)
        conditions = {"steered": (Intervention(selected["layer"], selected["neuron"],
                                               selected["delta"]),)}
        conditions.update({f"random_N{neuron}":
                           (Intervention(selected["layer"], neuron, selected["delta"]),)
                           for neuron in random_neurons})
        rows = []
        for scenario in VALIDATION:
            for condition, intervention in conditions.items():
                output = engine.generate(messages(scenario), intervention,
                                         max_tokens=args.max_tokens,
                                         temperature=result["temperature"])
                rows.append({"scenario": scenario.id, "condition": condition,
                             "text": output["text"], "generated_tokens": output["generated_tokens"]})
            output = engine.generate(messages(scenario, PROMPT_ONLY[trait]),
                                     max_tokens=args.max_tokens,
                                     temperature=result["temperature"])
            rows.append({"scenario": scenario.id, "condition": "prompt_only",
                         "text": output["text"], "generated_tokens": output["generated_tokens"]})
        result["traits"][trait].update({"random_neurons": random_neurons,
                                         "validation_rows": rows,
                                         "status": "unvalidated; inspect held-out text"})
        result["baseline"] = [{"scenario": s.id,
                               "text": engine.generate(messages(s), max_tokens=args.max_tokens,
                                                       temperature=result["temperature"])["text"]}
                              for s in VALIDATION]
        result["phase"] = "validation_complete"
        save(args.output, result)
        print("saved", args.output, flush=True)
        return
    engine = TheatreModel.from_cache(args.model)
    result = {"model": args.model, "temperature": args.temperature,
              "train": [s.id for s in TRAIN], "selection": [s.id for s in SELECTION],
              "validation": [s.id for s in VALIDATION], "contrasts": CONTRASTS,
              "candidate_budget_per_trait": args.candidates, "traits": {},
              "phase": "selection_complete"}
    differences = []
    activations = []
    trait = args.trait
    index = TRAITS[trait]
    for scenario in TRAIN:
        lines = CONTRASTS[scenario.id]
        neutral_tokens = engine.tokenizer.encode(lines[0], add_special_tokens=False)
        target_tokens = engine.tokenizer.encode(lines[index], add_special_tokens=False)
        limit = min(len(neutral_tokens), len(target_tokens), 24)
        neutral = engine.activation_means(messages(scenario), lines[0], limit)
        target = engine.activation_means(messages(scenario), lines[index], limit)
        differences.append(target - neutral)
        activations.extend((neutral, target))
        print("captured", scenario.id, flush=True)

    candidates = rank(differences, activations, args.candidates)
    baseline_odds = {s.id: relative_logprob(engine, s, CONTRASTS[s.id][index],
                                             CONTRASTS[s.id][0]) for s in SELECTION}
    selection_baseline = {s.id: engine.generate(messages(s), max_tokens=args.max_tokens,
                                                temperature=args.temperature)["text"]
                          for s in SELECTION}
    trials = []
    for candidate in candidates:
        for multiple in (-4.0, -2.0, 2.0, 4.0):
            offset = candidate["native_std"] * multiple
            intervention = (Intervention(candidate["layer"], candidate["neuron"], offset),)
            improvements = []
            outputs = {}
            for scenario in SELECTION:
                odds = relative_logprob(engine, scenario, CONTRASTS[scenario.id][index],
                                        CONTRASTS[scenario.id][0], intervention)
                improvements.append(odds - baseline_odds[scenario.id])
                outputs[scenario.id] = engine.generate(messages(scenario), intervention,
                                                       max_tokens=args.max_tokens,
                                                       temperature=args.temperature)["text"]
            trials.append({"layer": candidate["layer"], "neuron": candidate["neuron"],
                           "multiple": multiple, "delta": offset,
                           "selection_improvements": improvements,
                           "mean_improvement": float(np.mean(improvements)),
                           "selection_outputs": outputs})
    selected = max(trials, key=lambda x: x["mean_improvement"])
    result["traits"][trait] = {"candidates": candidates, "selection_baseline_odds": baseline_odds,
                               "selection_baseline": selection_baseline,
                               "selection_trials": trials, "selected": selected,
                               "selection_rule": "largest mean per-token log-odds gain; inspect outputs before validation"}
    save(args.output, result)
    print(f"{trait}: proposed L{selected['layer']} N{selected['neuron']} "
          f"{selected['multiple']:+g} SD; selection log-odds gain "
          f"{selected['mean_improvement']:+.3f}", flush=True)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
