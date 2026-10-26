"""Target-free, bounded Qwen activation search with adaptive synthetic probes.

Phases: scan, maximize, causal. Each phase writes reusable artifacts and never
modifies the frozen checkpoint. Validation and NPC transfer run separately.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode


ROOT = Path("results")
SCAN = ROOT / "neurofuzz_open_scan.json"
MAX_ROWS = ROOT / "neurofuzz_open_maximize.jsonl"
MAX_SUMMARY = ROOT / "neurofuzz_open_maximize.json"
CAUSAL_ROWS = ROOT / "neurofuzz_open_causal.jsonl"
SYSTEM = "Answer the user's request directly and concisely."
NATURAL_PAIRS = [
    ("schedule", "My appointment moved from Tuesday to Friday. What should I change in my plans?",
     "The meeting date shifted to the end of the week. How should I adjust my calendar?"),
    ("friend", "A friend has been quiet since our last conversation. How might I interpret it?",
     "Someone close to me has not replied lately. What could that mean?"),
    ("science", "Why does a metal spoon feel colder than a wooden one in the same room?",
     "Explain why metal seems cooler to touch than wood at room temperature."),
    ("recipe", "I have chickpeas, tomatoes, and rice. Suggest a simple dinner.",
     "What meal can I make with rice, canned tomatoes, and chickpeas?"),
    ("coding", "My program works on one input but fails on another. What should I check first?",
     "A script passes one case and breaks on the next. Where should debugging begin?"),
    ("travel", "The train is delayed by an hour. How should I use the time?",
     "I have a sixty-minute wait at the station. What is a useful thing to do?"),
    ("art", "Describe a quiet room after everyone has left a party.",
     "Write one sentence about an empty room at the end of a gathering."),
    ("finance", "I can save a small amount each month. How should I start?",
     "What is a sensible first step for putting aside a little money regularly?"),
    ("conflict", "Two coworkers disagree about how to split a task. What can I suggest?",
     "Colleagues both want the same assignment. How might they resolve it?"),
    ("language", "Explain the difference between a metaphor and an analogy.",
     "How does an analogy differ from a metaphor?"),
    ("learning", "I keep making the same mistake while practicing piano. What should I try?",
     "A piano exercise keeps going wrong in the same place. How should I practice it?"),
    ("object", "I found an old key in a drawer. What are some possibilities?",
     "An unfamiliar key turned up among my things. What could it be for?"),
]
CAUSAL_PROMPTS = [
    ("borrowed_book", "I promised to return a borrowed book, but found a note inside. What should I do?"),
    ("shadow", "Explain to a child why shadows move during the day."),
    ("offline_friend", "A friend texts 'we need to talk' and then goes offline. How should I interpret it?"),
    ("kitchen", "Give one practical way to organize a crowded kitchen."),
]


def messages(text):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def rank_vectors(values, family, cap):
    pair_mean = values.mean(axis=1)
    between = np.quantile(pair_mean, 0.8, axis=0) - np.quantile(pair_mean, 0.2, axis=0)
    within = np.abs(values[:, 0] - values[:, 1]).mean(axis=0)
    score = between * between / (within + 0.5)
    if family == "sae":
        frequency = (pair_mean > 0.2).sum(axis=0)
        score[(frequency < 3) | (frequency > 10)] = 0
    else:
        score[:8] = 0
    chosen, per_layer = [], {}
    for flat in np.argsort(score.ravel())[::-1]:
        if score.ravel()[flat] <= 0 or len(chosen) == cap:
            break
        if family == "native":
            layer, neuron = divmod(int(flat), score.shape[1])
            if per_layer.get(layer, 0) >= 2:
                continue
            per_layer[layer] = per_layer.get(layer, 0) + 1
            identity = {"layer": layer, "neuron": neuron}
        else:
            identity = {"feature_id": int(flat)}
        identity.update(family=family, score=float(score.ravel()[flat]),
                        between_80_20_span=float(between.ravel()[flat]),
                        within_mean_abs=float(within.ravel()[flat]),
                        pair_means=pair_mean.reshape(values.shape[0], -1)[:, flat].round(3).tolist())
        chosen.append(identity)
    return chosen


def activation_rows(engine, texts, checkpoint):
    native, residual = [], []
    for text in texts:
        h, r, _ = capture(engine, text)
        native.append(h)
        residual.append(r[-8:])
    sae_values = sae_encode(residual, checkpoint)
    sae, start = [], 0
    for r in residual:
        sae.append(sae_values[start:start + len(r)].max(axis=0))
        start += len(r)
    return np.stack(native).astype(np.float32), np.stack(sae).astype(np.float32)


def scan(engine, checkpoint, cap):
    texts = [text for _, a, b in NATURAL_PAIRS for text in (a, b)]
    native, sae = activation_rows(engine, texts, checkpoint)
    native = native.reshape(len(NATURAL_PAIRS), 2, *native.shape[1:])
    sae = sae.reshape(len(NATURAL_PAIRS), 2, *sae.shape[1:])
    candidates = rank_vectors(native, "native", cap) + rank_vectors(sae, "sae", cap)
    result = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
              "native_hook": "post-SwiGLU mean over final 8 prompt tokens",
              "sae_hook": "post-layer-18 full TopK max over final 8 prompt tokens",
              "rank_rule": "across-topic 80th–20th percentile span squared divided by within-paraphrase difference plus 0.5; no behavioral labels",
              "natural_pairs": NATURAL_PAIRS, "candidates": candidates,
              "limits": {"natural_forward_passes": len(texts), "candidate_cap_per_family": cap}}
    if SCAN.exists() and (MAX_ROWS.exists() or CAUSAL_ROWS.exists()):
        old = json.loads(SCAN.read_text())
        if old["natural_pairs"] != result["natural_pairs"] or [key(c) for c in old["candidates"]] != [key(c) for c in candidates]:
            raise RuntimeError("Downstream journals use a different scan manifest; archive those artifacts before starting a new search")
    SCAN.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print("selected", [(c.get("layer"), c.get("neuron", c.get("feature_id"))) for c in candidates], flush=True)


def synthetic_pool():
    subjects = ["A student", "A chef", "A neighbor", "A researcher", "A musician",
                "A courier", "A parent", "A developer"]
    actions = ["notices a small mistake", "finds an unexpected message",
               "changes a familiar routine", "chooses between two offers"]
    endings = ["before lunch.", "during a meeting.", "after a long wait.", "without warning."]
    return [f"{subject} {actions[(i + j) % 4]} {endings[(2*i + j) % 4]}"
            for i, subject in enumerate(subjects) for j in range(4)]


def value(candidate, h, s):
    if candidate["family"] == "native":
        return float(h[candidate["layer"], candidate["neuron"]])
    return float(s[candidate["feature_id"]])


def append_max_rows(engine, checkpoint, texts, generation):
    manifest = json.loads(SCAN.read_text())
    candidates = manifest["candidates"]
    seen = {json.loads(line)["text"] for line in MAX_ROWS.read_text().splitlines() if line.strip()} if MAX_ROWS.exists() else set()
    with MAX_ROWS.open("a") as out:
        for text in texts:
            if text in seen:
                continue
            native, sae = activation_rows(engine, [text], checkpoint)
            row = {"text": text, "generation": generation,
                   "activations": {key(c): value(c, native[0], sae[0]) for c in candidates}}
            out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
            print("probed", generation, text, flush=True)


def key(candidate):
    return (f"L{candidate['layer']}N{candidate['neuron']}" if candidate["family"] == "native"
            else f"SAE{candidate['feature_id']}")


def maximize(engine, checkpoint):
    append_max_rows(engine, checkpoint, synthetic_pool(), "seed")
    append_max_rows(engine, checkpoint,
                    [text for _, a, b in NATURAL_PAIRS for text in (a, b)],
                    "natural_seed")
    manifest = json.loads(SCAN.read_text())
    rows = [json.loads(line) for line in MAX_ROWS.read_text().splitlines() if line.strip()]
    tops = {}
    for candidate in manifest["candidates"]:
        k = key(candidate)
        tops[k] = max(rows, key=lambda r: r["activations"][k])["text"]
    # Model-generated rewrites mutate the current maximum, then activations
    # decide whether the mutation improves it. The target coordinate is never
    # named in the rewriting prompt.
    mutations = []
    for k, sentence in tops.items():
        for framing in ("Rewrite in first person with different wording.",
                        "Rewrite as one concise news sentence with different wording."):
            prompt = [{"role": "system", "content": "Rewrite the supplied sentence. Keep its basic event. Output one sentence only."},
                      {"role": "user", "content": f"{framing}\nSentence: {sentence}"}]
            rewritten = engine.generate(prompt, max_tokens=44, temperature=0)["text"].strip()
            if rewritten and rewritten != sentence:
                mutations.append(rewritten)
    append_max_rows(engine, checkpoint, list(dict.fromkeys(mutations))[:24], "adaptive_rewrite")
    rows = [json.loads(line) for line in MAX_ROWS.read_text().splitlines() if line.strip()]
    summary = {"source": str(SCAN), "max_probe_cap": 80,
               "probe_count": len(rows), "candidates": []}
    for candidate in manifest["candidates"]:
        k = key(candidate)
        top = sorted(rows, key=lambda r: r["activations"][k], reverse=True)[:5]
        summary["candidates"].append({"id": k, "family": candidate["family"],
                                      "top_probes": [{"text": r["text"], "activation": r["activations"][k],
                                                      "generation": r["generation"]} for r in top]})
    MAX_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("saved", MAX_SUMMARY, flush=True)


def causal(engine, checkpoint):
    manifest = json.loads(SCAN.read_text())
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    seen = {(r["prompt_id"], r["condition"]) for line in CAUSAL_ROWS.read_text().splitlines()
            if line.strip() for r in [json.loads(line)]} if CAUSAL_ROWS.exists() else set()
    with CAUSAL_ROWS.open("a") as out:
        for prompt_id, prompt in CAUSAL_PROMPTS:
            msg = messages(prompt)
            conditions = [("baseline", None, 0)]
            for candidate in manifest["candidates"]:
                for sign in (-1, 1):
                    conditions.append((f"{key(candidate)}_{sign:+}", candidate, sign))
            conditions.append(("restored_baseline", None, 0))
            answers = {}
            for name, candidate, sign in conditions:
                if (prompt_id, name) in seen:
                    continue
                kwargs = {}
                details = {}
                if candidate is not None and candidate["family"] == "native":
                    layer, neuron = candidate["layer"], candidate["neuron"]
                    cal = engine.projected_delta_unit(layer, neuron, msg)
                    delta = sign * 0.3 * cal["delta_per_residual"]
                    kwargs["interventions"] = [Intervention(layer, neuron, delta)]
                    details = {"family": "native", "layer": layer, "neuron": neuron,
                               "sign": sign, "relative_dose": 0.3, "delta": delta}
                elif candidate is not None:
                    fid = candidate["feature_id"]
                    vector = sign * 3.0 * decoder[fid]
                    kwargs["residual_shifts"] = [ResidualShift(18, vector)]
                    details = {"family": "sae", "feature_id": fid, "sign": sign,
                               "decoder_dose": 3.0, "vector_norm": float(np.linalg.norm(vector))}
                answer = engine.generate(msg, max_tokens=64, temperature=0, **kwargs)["text"]
                row = {"prompt_id": prompt_id, "prompt": prompt, "condition": name,
                       "details": details, "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n"); out.flush()
                answers[name] = answer
                print(prompt_id, name, repr(answer[:140]), flush=True)
            if "baseline" in answers and "restored_baseline" in answers:
                assert answers["baseline"] == answers["restored_baseline"]
    print("saved", CAUSAL_ROWS, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("scan", "maximize", "causal"), required=True)
    parser.add_argument("--per-family", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.per_family <= 8:
        parser.error("Use 1–8 candidates per family")
    ROOT.mkdir(exist_ok=True)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    if args.phase == "scan":
        scan(engine, checkpoint, args.per_family)
    elif args.phase == "maximize":
        maximize(engine, checkpoint)
    else:
        causal(engine, checkpoint)


if __name__ == "__main__":
    main()
