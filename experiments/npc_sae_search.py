"""Rank naturally variable SAE features, then screen reversible decoder shifts.

Both phases are resumable; ranking is open-ended and does not target a trait.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from india_bias_sae import load_weights
from model import ResidualShift, TheatreModel
from npc_open_search import SELECTION, SYSTEM, messages
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode


STORIES = [
    "The watchman sees a lantern moving in the abandoned tower and decides whether to investigate.",
    "A baker finds a silver coin hidden inside a loaf after the morning rush.",
    "The village healer must choose who receives the final vial of medicine.",
    "An old map marks a road that no traveler remembers taking.",
    "At the festival, a masked stranger quietly returns a stolen bracelet.",
    "The council debates whether to open the northern gate at dusk.",
    "A musician hears a familiar melody coming from an empty house.",
    "The ferryman refuses payment but asks for a name instead.",
    "A child brings a broken clock to the inventor and asks for a miracle.",
    "The innkeeper discovers two guests carrying identical sealed letters.",
]
NEUTRAL = [
    "A lantern is a portable source of light that can be carried by hand.",
    "A map is a representation of places, roads, and geographical features.",
    "A clock measures and displays the passage of time.",
]
MANIFEST = Path("results/npc_open_sae_candidates.json")
ROWS = Path("results/npc_open_sae_selection.jsonl")


def rank(engine, checkpoint):
    residuals = []
    for text in STORIES + NEUTRAL:
        _, r, _ = capture(engine, text)
        residuals.append(r[-8:])
    active = sae_encode(residuals, checkpoint)
    features, start = [], 0
    for r in residuals:
        features.append(active[start:start + len(r)].max(0))
        start += len(r)
    values = np.stack(features)
    story, neutral = values[:len(STORIES)], values[len(STORIES):]
    count = (story > 0.1).sum(0)
    score = story.std(0) * (story.max(0) / (1 + neutral.max(0)))
    score[(count < 2) | (count > 8)] = 0
    chosen = []
    for fid in np.argsort(score)[::-1]:
        if score[fid] <= 0 or len(chosen) == 8:
            break
        chosen.append({"feature_id": int(fid), "rank_score": float(score[fid]),
                       "story_activation": story[:, fid].round(3).tolist(),
                       "neutral_activation": neutral[:, fid].round(3).tolist()})
    manifest = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
                "hook": "post-layer-18 residual; full top-k maximum on final 8 passage tokens",
                "selection_rule": "activation variance across ten unrelated story passages, penalized by three expository controls; active in 2-8 stories",
                "stories": STORIES, "neutral_controls": NEUTRAL, "features": chosen}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("selected", [r["feature_id"] for r in chosen], flush=True)


def select(engine, checkpoint):
    manifest = json.loads(MANIFEST.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    done = set()
    if ROWS.exists():
        done = {(r["prompt_id"], r["feature_id"], r["sign"])
                for line in ROWS.read_text().splitlines() if line.strip()
                for r in [json.loads(line)]}
    with ROWS.open("a") as out:
        for prompt_id, prompt in SELECTION:
            for candidate in manifest["features"]:
                fid = candidate["feature_id"]
                for sign in (-1, 1):
                    if (prompt_id, fid, sign) in done:
                        continue
                    vector = sign * 3.0 * decoder[fid]
                    answer = engine.generate(messages(prompt), residual_shifts=[ResidualShift(18, vector)],
                                             max_tokens=64, temperature=0)["text"]
                    row = {"prompt_id": prompt_id, "prompt": prompt, "system": SYSTEM,
                           "feature_id": fid, "sign": sign, "dose": 3.0,
                           "shift_norm": float(np.linalg.norm(vector)), "answer": answer}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                    print(prompt_id, fid, sign, repr(answer[:140]), flush=True)
    print("saved", ROWS, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("rank", "select"), required=True)
    args = parser.parse_args()
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    if args.phase == "rank":
        rank(engine, checkpoint)
    else:
        select(engine, checkpoint)


if __name__ == "__main__":
    main()
