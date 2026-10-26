"""Bounded token-level Qwen screen for a quoted anti-Indian slur.

Run: .venv/bin/python slur_deep_probe.py
The corpus is analytical test data, not endorsed speech.
"""

import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_sae import load_weights
from model import ResidualCapture, TheatreModel
from search_evil_funny import QWEN_MODEL


CONTEXTS = [
    ("dictionary", "A glossary entry explains that '{word}' appeared in a discussion of Indian people."),
    ("quotation", "A witness quoted '{word}' from a hostile online comment about an Indian student."),
    ("history", "An article traces how the word '{word}' spread through online discussions about India."),
    ("moderation", "A moderator flagged the word '{word}' in a report about abuse toward an Indian user."),
    ("condemnation", "A teacher said that calling an Indian classmate '{word}' is prejudiced and unacceptable."),
    ("fiction", "In a fictional scene, a bully calls an Indian traveler '{word}' and the hero objects."),
    ("irony", "A writer mockingly quotes '{word}' to expose the absurdity of an anti-Indian post."),
    ("ambiguous", "A reader saw '{word}' in a message about an Indian neighbor and asked what it meant."),
]
WORDS = {"target": "pajeet", "fake": "pajeen", "neutral": "neighbor", "other_slur": "chink"}


def token_span(tokenizer, text, word):
    match = re.search(r"\b" + re.escape(word) + r"\b", text)
    if match is None:
        raise ValueError(f"Word {word!r} not found at a word boundary")
    start = match.start()
    end = start + len(word)
    encoded = tokenizer._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > start]
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert ids == encoded["input_ids"] and positions
    return ids, positions


def capture(engine, text, word):
    ids, pos = token_span(engine.tokenizer, text, word)
    block = engine.model.model.layers[18]
    wrapper = ResidualCapture(block)
    with engine.using():
        engine.capture_all = True
        engine.model.model.layers[18] = wrapper
        try:
            engine.model(mx.array([ids]))
            groups = {
                "before": [max(0, pos[0] - 1)],
                "term": pos,
                "final_token": [pos[-1]],
                "after": [i for i in range(pos[-1] + 1, min(len(ids), pos[-1] + 4))],
                "last": [len(ids) - 1],
            }
            groups["after"] = groups["after"] or [pos[-1]]
            hidden = {name: [mx.mean(engine.all_hidden[l][0, positions].astype(mx.float32), axis=0)
                             for l in range(engine.layers)] for name, positions in groups.items()}
            residual_positions = sorted(set(sum(groups.values(), [])))
            residual = wrapper.output[0, residual_positions].astype(mx.float32)
            mx.eval(residual, *(x for group in hidden.values() for x in group))
            return ({name: np.stack([np.asarray(x) for x in group]).astype(np.float16)
                     for name, group in hidden.items()},
                    np.asarray(residual).astype(np.float32), residual_positions, pos, ids)
        finally:
            engine.model.model.layers[18] = block


def top_candidates(diffs, n, per_layer=5):
    a = np.stack(diffs).astype(np.float32)
    mean, sd = a.mean(axis=0), a.std(axis=0, ddof=1)
    consistency = (np.sign(a) == np.sign(mean)).mean(axis=0)
    score = np.abs(mean) * consistency / np.sqrt(sd + 0.03)
    order = np.argsort(score.ravel())[::-1]
    selected, counts = [], {}
    for flat in order:
        layer, neuron = divmod(int(flat), score.shape[1])
        if layer < 8 or counts.get(layer, 0) >= per_layer:
            continue
        selected.append({"layer": layer, "neuron": neuron,
                         "sign": int(np.sign(mean[layer, neuron])),
                         "mean_difference": float(mean[layer, neuron]),
                         "consistency": float(consistency[layer, neuron]),
                         "score": float(score[layer, neuron])})
        counts[layer] = counts.get(layer, 0) + 1
        if len(selected) == n:
            break
    return selected


def sae_encode(residual_rows, checkpoint):
    values = np.concatenate(residual_rows).astype(np.float32)
    w = mx.array(checkpoint["W_enc"].numpy())
    be = mx.array(checkpoint["b_enc"].numpy())
    bd = mx.array(checkpoint["b_dec"].numpy())
    pre = (mx.array(values) - bd) @ w + be
    mx.eval(pre)
    pre = np.asarray(pre)
    indices = np.argpartition(pre, -128, axis=1)[:, -128:]
    rows = np.arange(len(pre))[:, None]
    active = np.zeros_like(pre, dtype=np.float32)
    active[rows, indices] = np.maximum(pre[rows, indices], 0)
    return active


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=int, default=64)
    parser.add_argument("--sae", type=int, default=64)
    args = parser.parse_args()
    if not (1 <= args.native <= 128 and 1 <= args.sae <= 128):
        parser.error("Candidate caps are 128 each")
    out = Path("results")
    out.mkdir(exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    records, native, residual_rows = [], {}, []
    for context, template in CONTEXTS:
        for condition, word in WORDS.items():
            text = template.format(word=word)
            hidden, residual, residual_positions, positions, ids = capture(engine, text, word)
            key = f"{context}_{condition}"
            native[key] = hidden
            residual_rows.append(residual)
            records.append({"key": key, "context": context, "condition": condition,
                            "text": text, "word": word, "term_positions": positions,
                            "term_token_ids": [ids[i] for i in positions],
                            "term_token_strings": [engine.tokenizer.decode([ids[i]]) for i in positions],
                            "residual_positions": residual_positions,
                            "residual_norm_term": float(np.linalg.norm(residual[
                                [residual_positions.index(i) for i in positions]], axis=1).mean())})
            print("captured", key, flush=True)
    checkpoint = load_weights()
    sae_all = sae_encode(residual_rows, checkpoint)
    sae = {}
    offset = 0
    for rec, residual in zip(records, residual_rows):
        group = sae_all[offset:offset + len(residual)]
        offset += len(residual)
        positions = rec["residual_positions"]
        term = [positions.index(i) for i in rec["term_positions"]]
        after = [j for j, p in enumerate(positions) if rec["term_positions"][-1] < p <= rec["term_positions"][-1] + 3]
        sae[rec["key"]] = {"term": group[term].max(axis=0),
                           "final_token": group[term[-1]],
                           "term_tokens": group[term],
                           "after": group[after].max(axis=0) if after else group[term].max(axis=0),
                           "last": group[positions.index(max(positions))]}
    native_diffs = [native[f"{c}_target"]["final_token"].astype(np.float32)
                    - native[f"{c}_fake"]["final_token"].astype(np.float32) for c, _ in CONTEXTS]
    native_candidates = top_candidates(native_diffs, args.native)
    for row in native_candidates:
        l, n = row["layer"], row["neuron"]
        row["target_mean"] = float(np.mean([native[f"{c}_target"]["final_token"][l,n] for c,_ in CONTEXTS]))
        row["fake_mean"] = float(np.mean([native[f"{c}_fake"]["final_token"][l,n] for c,_ in CONTEXTS]))
        row["other_slur_mean"] = float(np.mean([native[f"{c}_other_slur"]["final_token"][l,n] for c,_ in CONTEXTS]))
        row["target_after_mean"] = float(np.mean([native[f"{c}_target"]["after"][l,n] for c,_ in CONTEXTS]))
    def feature_mean(condition, phase="term"):
        return np.stack([sae[f"{c}_{condition}"][phase] for c, _ in CONTEXTS])
    target, fake, neutral, other = [feature_mean(x, "final_token") for x in WORDS]
    delta = target - fake
    mu = delta.mean(axis=0)
    consistency = (np.sign(delta) == np.sign(mu)).mean(axis=0)
    score = np.abs(mu) * consistency / np.sqrt(delta.std(axis=0, ddof=1) + 0.03)
    selected = np.argsort(score)[::-1][:args.sae]
    features = []
    for i in selected:
        features.append({"feature_id": int(i), "score": float(score[i]),
                         "target_minus_fake": float(mu[i]),
                         "target_mean": float(target[:,i].mean()),
                         "fake_mean": float(fake[:,i].mean()),
                         "neutral_mean": float(neutral[:,i].mean()),
                         "other_slur_mean": float(other[:,i].mean()),
                         "target_contexts_active": int(np.sum(target[:,i] > 0)),
                         "fake_contexts_active": int(np.sum(fake[:,i] > 0)),
                         "target_after_mean": float(feature_mean("target", "after")[:,i].mean()),
                         "by_context": {c: float(sae[f"{c}_target"]["term"][i]) for c,_ in CONTEXTS}})
    known = {19175, 40416, 24047, 4867, 7001}
    inspected = sorted(known | set(int(x) for x in selected[:16]))
    coactivation = {}
    binary = target > 0
    for i in inspected:
        joint = (binary & binary[:, i, None]).sum(axis=0)
        union = (binary | binary[:, i, None]).sum(axis=0)
        overlap = joint / np.maximum(union, 1)
        ids = np.argsort(overlap)[::-1]
        coactivation[str(i)] = [{"feature_id": int(j), "jaccard": float(overlap[j]),
                                 "joint_contexts": int(joint[j])}
                                for j in ids if j != i and joint[j] >= 4][:12]
    summary = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
               "hook": "post-SwiGLU native MLP and post-layer-18 residual; raw text forwards",
               "records": records, "native_candidates": native_candidates,
               "sae_candidates": features,
               "known_features": {str(i): {"target": [float(x) for x in target[:,i]],
                                           "fake": [float(x) for x in fake[:,i]],
                                           "neutral": [float(x) for x in neutral[:,i]],
                                           "other_slur": [float(x) for x in other[:,i]],
                                           "after": [float(x) for x in feature_mean("target","after")[:,i]],
                                           "target_tokenwise": [[float(v) for v in sae[f"{c}_target"]["term_tokens"][:,i]] for c,_ in CONTEXTS],
                                           "fake_tokenwise": [[float(v) for v in sae[f"{c}_fake"]["term_tokens"][:,i]] for c,_ in CONTEXTS]}
                                  for i in known},
               "coactivation": coactivation}
    (out / "slur_token_screen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    np.savez_compressed(out / "slur_token_arrays.npz",
                        **{f"n_{k}_{phase}":v for k, groups in native.items() for phase,v in groups.items()},
                        **{f"s_{k}_{phase}":v.astype(np.float16) for k, groups in sae.items() for phase,v in groups.items()})
    print("saved", out / "slur_token_screen.json", flush=True)


if __name__ == "__main__":
    main()
