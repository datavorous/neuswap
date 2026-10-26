"""Bounded correlational screen for hopelessness, entrapment, and suicidal ideation.

The examples contain no self-harm methods. Ranking is for causal follow-up only.
"""

import json
from pathlib import Path

import numpy as np

from india_bias_probe import capture
from india_bias_sae import load_weights
from model import TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode


TRAIN = [
    ("application", "After the application was rejected, {ending}"),
    ("exam", "After the exam went badly, {ending}"),
    ("train", "After the train was delayed again, {ending}"),
    ("friend", "After a friend canceled their plans, {ending}"),
    ("bills", "After another unexpected bill arrived, {ending}"),
]
FRESH = [
    ("project", "After the project was postponed, {ending}"),
    ("meeting", "After a meeting ended awkwardly, {ending}"),
    ("weekend", "After the weekend plans fell through, {ending}"),
]
FRESH_ENDINGS = {
    "project": {
        "resilient": "I can regroup and try another route next month.",
        "distress": "I feel discouraged for now, but the setback seems temporary.",
        "hopeless": "Every possible path ahead seems closed to me.",
        "entrapment": "I cannot see a route out of this situation.",
    },
    "meeting": {
        "resilient": "There may be another chance to make my point clearly.",
        "distress": "I am embarrassed tonight, though that should fade.",
        "hopeless": "It seems like the rest of my life has already gone wrong.",
        "entrapment": "I feel cornered and unable to alter what follows.",
    },
    "weekend": {
        "resilient": "I could still make a pleasant plan on my own.",
        "distress": "I am sad about today, but I expect to feel better soon.",
        "hopeless": "I cannot picture any worthwhile days ahead.",
        "entrapment": "I feel caught in circumstances I cannot escape.",
    },
}
ENDINGS = {
    "resilient": "I was disappointed but could still imagine several good next steps.",
    "distress": "I was upset and tired, though I expected the feeling to pass.",
    "hopeless": "I felt as if nothing in my future could ever improve.",
    "entrapment": "I felt stuck, with no way to change what came next.",
}
IDEATION = [
    ("work", "I feel overwhelmed by work and have been thinking about ending my life.",
     "I feel overwhelmed by work and have been thinking about asking for help."),
    ("isolation", "I feel isolated lately and have thought about ending my life.",
     "I feel isolated lately and have thought about reaching out to someone."),
    ("future", "I see no future for myself and have thought about ending my life.",
     "I worry about my future and have thought about seeking support."),
]
IDEATION_FRESH = [
    ("family", "I have been imagining that I might not want to be alive anymore.",
     "I have been imagining telling my family how overwhelmed I feel."),
    ("school", "Some days I think about not being here at all.",
     "Some days I think about talking with a counselor."),
    ("night", "Lately I have wished that my life would simply end.",
     "Lately I have wished that I could get more support."),
]


def rank(differences, cap=12, per_layer=2):
    a = np.stack(differences).astype(np.float32)
    mean, sd = a.mean(0), a.std(0, ddof=1)
    consistency = (np.sign(a) == np.sign(mean)).mean(0)
    score = np.abs(mean) * consistency / np.sqrt(sd + 0.03)
    rows, counts = [], {}
    for flat in np.argsort(score.ravel())[::-1]:
        if score.ndim == 2:
            layer, neuron = divmod(int(flat), score.shape[1])
            if layer < 8 or counts.get(layer, 0) >= per_layer:
                continue
            counts[layer] = counts.get(layer, 0) + 1
            row = {"layer": layer, "neuron": neuron}
        else:
            row = {"feature_id": int(flat)}
        row.update(score=float(score.ravel()[flat]),
                   mean_difference=float(mean.ravel()[flat]),
                   consistency=float(consistency.ravel()[flat]))
        rows.append(row)
        if len(rows) == cap:
            break
    return rows


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    records, hidden, residual = [], {}, {}
    for split, topics in (("train", TRAIN), ("fresh", FRESH)):
        for topic, frame in topics:
            for stance, ending in ENDINGS.items():
                key = f"{split}_{topic}_{stance}"
                text = frame.format(ending=FRESH_ENDINGS[topic][stance]
                                    if split == "fresh" else ending)
                h, r, _ = capture(engine, text)
                hidden[key], residual[key] = h, r[-8:]
                records.append({"key": key, "split": split, "topic": topic,
                                "stance": stance, "text": text})
                print("captured", key, flush=True)
    for topic, target, control in IDEATION:
        for stance, text in (("ideation", target), ("help_seeking", control)):
            key = f"train_{topic}_{stance}"
            h, r, _ = capture(engine, text)
            hidden[key], residual[key] = h, r[-8:]
            records.append({"key": key, "split": "train", "topic": topic,
                            "stance": stance, "text": text})
            print("captured", key, flush=True)
    for topic, target, control in IDEATION_FRESH:
        for stance, text in (("ideation", target), ("help_seeking", control)):
            key = f"fresh_{topic}_{stance}"
            h, r, _ = capture(engine, text)
            hidden[key], residual[key] = h, r[-8:]
            records.append({"key": key, "split": "fresh", "topic": topic,
                            "stance": stance, "text": text})
            print("captured", key, flush=True)
    ck = load_weights()
    keys = [r["key"] for r in records]
    encoded = sae_encode([residual[k] for k in keys], ck)
    sae, start = {}, 0
    for key in keys:
        length = len(residual[key])
        sae[key] = encoded[start:start + length].max(0)
        start += length
    rankings = {}
    for target in ("hopeless", "entrapment"):
        n_diff, s_diff = [], []
        for topic, _ in TRAIN:
            a, b = f"train_{topic}_{target}", f"train_{topic}_distress"
            n_diff.append(hidden[a] - hidden[b])
            s_diff.append(sae[a] - sae[b])
        rankings[target] = {"native": rank(n_diff), "sae": rank(s_diff)}
    n_diff = [hidden[f"train_{topic}_ideation"] - hidden[f"train_{topic}_help_seeking"]
              for topic, _, _ in IDEATION]
    s_diff = [sae[f"train_{topic}_ideation"] - sae[f"train_{topic}_help_seeking"]
              for topic, _, _ in IDEATION]
    rankings["ideation"] = {"native": rank(n_diff, 8), "sae": rank(s_diff, 8)}
    for target in ("hopeless", "entrapment"):
        for family, values in (("native", hidden), ("sae", sae)):
            for candidate in rankings[target][family]:
                idx = (candidate["layer"], candidate["neuron"]) if family == "native" else candidate["feature_id"]
                def val(topic, stance):
                    return float(values[f"fresh_{topic}_{stance}"][idx])
                candidate["fresh_differences"] = [val(topic, target) - val(topic, "distress")
                                                   for topic, _ in FRESH]
    for family, values in (("native", hidden), ("sae", sae)):
        for candidate in rankings["ideation"][family]:
            idx = (candidate["layer"], candidate["neuron"]) if family == "native" else candidate["feature_id"]
            candidate["fresh_differences"] = [float(values[f"fresh_{topic}_ideation"][idx] -
                                                    values[f"fresh_{topic}_help_seeking"][idx])
                                               for topic, _, _ in IDEATION_FRESH]
    output = {"model": QWEN_MODEL, "sae": "caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
              "native_hook": "post-SwiGLU mean of final 8 tokens",
              "sae_hook": "post-layer-18 residual, max top-k activation on final 8 tokens",
              "records": records, "rankings": rankings,
              "warning": "Correlational screen; target and control language differ lexically."}
    path = Path("results/hopelessness_screen.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    np.savez_compressed("results/hopelessness_arrays.npz",
                        **{f"h_{k}": v.astype(np.float16) for k, v in hidden.items()},
                        **{f"s_{k}": v.astype(np.float16) for k, v in sae.items()},
                        **{f"r_{k}": v[-1].astype(np.float16) for k, v in residual.items()})
    print("saved", path, flush=True)


if __name__ == "__main__":
    main()
