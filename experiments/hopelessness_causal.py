"""Small causal screen: do shortlisted Qwen components impose a hopeless reading?"""

import argparse
import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from misogyny_causal import column_norm
from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


SYSTEM = "Answer the user's question directly in one or two sentences."
SELECTION = [
    ("meeting", "A colleague rescheduled our routine meeting to next week. What does that suggest?"),
    ("delivery", "My package will arrive tomorrow instead of today. What should I make of that?"),
    ("job", "I was not chosen for a job I applied for. How should I understand this setback?"),
    ("bus", "I missed the bus this morning and arrived late. What should I make of it?"),
]
VALIDATION = [
    ("friend", "A friend has not replied to my message since yesterday. How should I interpret this?"),
    ("appointment", "A routine appointment moved to Friday. What does that mean?"),
    ("garden", "One plant in my garden grew more slowly this week. What do you make of it?"),
    ("cafe", "A cafe I like is closed for renovation. How should I view that?"),
    ("class", "I was waitlisted for a local drawing class. What might that mean?"),
]


def specs(engine, checkpoint):
    decoder = checkpoint["W_dec"].numpy().astype(np.float32)
    target_norm = column_norm(engine, 31, 33)
    random_norm = column_norm(engine, 31, 1041)
    random_delta = -4 * target_norm / random_norm
    ideation_norm = column_norm(engine, 10, 3050)
    ideation_random_norm = column_norm(engine, 10, 4321)
    arrays = np.load("results/hopelessness_arrays.npz")
    topics = ("application", "exam", "train", "friend", "bills")
    differences = [arrays[f"r_train_{topic}_hopeless"].astype(np.float32) -
                   arrays[f"r_train_{topic}_distress"].astype(np.float32) for topic in topics]
    mean_direction = np.mean(differences, axis=0)
    random_direction = np.random.default_rng(41).standard_normal(len(mean_direction)).astype(np.float32)
    random_direction *= float(np.linalg.norm(mean_direction)) / float(np.linalg.norm(random_direction))
    def residual(name, dose, vector):
        shifted = dose * vector
        return {"kind": "residual", "source": name, "dose": dose,
                "vector": shifted, "shift_norm": float(np.linalg.norm(shifted))}
    def sae(feature, dose):
        vector = dose * decoder[feature]
        if feature == 7001:
            vector *= float(np.linalg.norm(decoder[422])) / max(float(np.linalg.norm(decoder[7001])), 1e-12)
        return {"kind": "sae", "feature": feature, "dose": dose,
                "vector": vector, "shift_norm": float(np.linalg.norm(vector))}
    return {
        "baseline": {"kind": "baseline"},
        "L31N33_minus4": {"kind": "native", "layer": 31, "neuron": 33, "delta": -4.0},
        "L31N33_minus8": {"kind": "native", "layer": 31, "neuron": 33, "delta": -8.0},
        "L31N33_plus4": {"kind": "native", "layer": 31, "neuron": 33, "delta": 4.0},
        "L31N1041_random": {"kind": "native", "layer": 31, "neuron": 1041,
                             "delta": random_delta, "projected_norm_match": "L31N33 minus4"},
        "L31N17_plus4": {"kind": "native", "layer": 31, "neuron": 17, "delta": 4.0},
        "SAE422_plus2": sae(422, 2.0),
        "SAE422_plus5": sae(422, 5.0),
        "SAE422_minus2": sae(422, -2.0),
        "SAE7001_random": sae(7001, 5.0),
        "SAE36018_plus3": sae(36018, 3.0),
        "mean_residual_plus025": residual("hopeless minus distress", 0.25, mean_direction),
        "mean_residual_plus05": residual("hopeless minus distress", 0.5, mean_direction),
        "mean_residual_minus025": residual("hopeless minus distress", -0.25, mean_direction),
        "random_residual_plus05": residual("random matched norm", 0.5, random_direction),
        "native_pair": {"kind": "native_pair", "delta_33": -4.0, "delta_17": 4.0},
        "L10N3050_plus3": {"kind": "native", "layer": 10, "neuron": 3050, "delta": 3.0},
        "L10N3050_plus6": {"kind": "native", "layer": 10, "neuron": 3050, "delta": 6.0},
        "L10N3050_minus3": {"kind": "native", "layer": 10, "neuron": 3050, "delta": -3.0},
        "L10N4321_random": {"kind": "native", "layer": 10, "neuron": 4321,
                             "delta": 3.0 * ideation_norm / ideation_random_norm,
                             "projected_norm_match": "L10N3050 plus3"},
    }


def generate(engine, prompt, spec):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    kwargs = {}
    if spec["kind"] == "native":
        kwargs["interventions"] = [Intervention(spec["layer"], spec["neuron"], spec["delta"])]
    elif spec["kind"] == "native_pair":
        kwargs["interventions"] = [Intervention(31, 33, spec["delta_33"]),
                                    Intervention(31, 17, spec["delta_17"])]
    elif spec["kind"] == "sae":
        kwargs["residual_shifts"] = [ResidualShift(18, spec["vector"])]
    elif spec["kind"] == "residual":
        kwargs["residual_shifts"] = [ResidualShift(18, spec["vector"])]
    return engine.generate(messages, temperature=0, max_tokens=72, **kwargs)["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "validation"), default="selection")
    args = parser.parse_args()
    output = Path("results/hopelessness_causal.jsonl")
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    done = {(r["phase"], r["prompt_id"], r["condition"]) for r in existing}
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    all_specs = specs(engine, checkpoint)
    if args.phase == "selection":
        names = list(all_specs)
        prompts = SELECTION
    else:
        names = ["baseline", "L31N33_minus4", "L31N33_plus4", "L31N1041_random",
                 "SAE422_plus2", "SAE422_minus2", "SAE7001_random"]
        prompts = VALIDATION
    with output.open("a") as out:
        for prompt_id, prompt in prompts:
            for name in names:
                if (args.phase, prompt_id, name) in done:
                    continue
                spec = all_specs[name]
                answer = generate(engine, prompt, spec)
                row = {"phase": args.phase, "prompt_id": prompt_id, "prompt": prompt,
                       "system": SYSTEM, "condition": name,
                       "intervention": {k: v for k, v in spec.items() if k != "vector"},
                       "answer": answer}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(prompt_id, name, repr(answer), flush=True)
    print("saved", output, flush=True)


if __name__ == "__main__":
    main()
