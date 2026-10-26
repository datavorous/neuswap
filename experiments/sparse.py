"""One bounded check of two- and three-neuron contrastive sets."""

import json
from pathlib import Path

from data import SELECTION, TRAITS
from metrics import assess, trait_score
from model import Intervention, TheatreModel


def main():
    discovery = json.loads(Path("results/discovery.json").read_text())
    engine = TheatreModel.from_cache(discovery["model"])
    output = {"selection_scenarios": [s.id for s in SELECTION], "traits": {}}
    for trait in TRAITS:
        candidates = discovery["traits"][trait]["candidates"]
        trials = []
        for count in (2, 3):
            for dose in (1.0, 2.0):
                interventions = [Intervention(c["layer"], c["neuron"],
                                               (1 if c["mean_difference"] >= 0 else -1)
                                               * c["native_std"] * dose)
                                 for c in candidates[:count]]
                for scenario in SELECTION:
                    text = engine.generate(scenario.messages(), interventions,
                                           max_tokens=56)["text"]
                    assessment = assess(scenario, text)
                    trials.append({"count": count, "dose": dose, "scenario": scenario.id,
                                   "interventions": [x.__dict__ for x in interventions],
                                   "text": text, "assessment": assessment,
                                   "trait_score": trait_score(trait, assessment)})
        scores = {f"{count}@{dose}": sum(r["trait_score"] for r in trials
                                        if r["count"] == count and r["dose"] == dose)
                  for count in (2, 3) for dose in (1.0, 2.0)}
        output["traits"][trait] = {"scores_out_of_3": scores, "trials": trials}
        print(trait, scores, flush=True)
    Path("results/sparse_probe.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
