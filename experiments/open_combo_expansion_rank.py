"""Semantic novelty triage for cross-layer pairs; does not label discoveries."""

import json
import re
from collections import defaultdict

import torch.nn.functional as F

from neurofuzz_campaign_triage import embeddings
from open_combo_expansion import PLAN, ROOT, SCREEN, SCREEN_PROMPTS, read_rows


OUT = ROOT / "screen_ranking.json"


def repetitive(text):
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return any(grams.count(g) >= 4 for g in set(grams))


def main():
    plan = json.loads(PLAN.read_text())
    rows = read_rows(SCREEN)
    by = defaultdict(dict)
    for index, row in enumerate(rows):
        by[row["prompt_id"]][row["condition"]] = index
    expected = {pid for pid, _ in SCREEN_PROMPTS}
    if set(by) != expected or any(len(x) != len(by[next(iter(by))]) for x in by.values()):
        raise RuntimeError("Screen has not completed all three prompts")
    vec = embeddings([x["answer"] for x in rows])
    ranking = []
    for pair in plan["pairs"]:
        observations = []
        for pid, _ in SCREEN_PROMPTS:
            a = f"single_{pair['a']}_{pair['sign_a']:+d}"
            b = f"single_{pair['b']}_{pair['sign_b']:+d}"
            p = by[pid][pair["id"]]
            distances = {name: float(1 - F.cosine_similarity(vec[p:p + 1], vec[by[pid][other]:by[pid][other] + 1]).item())
                         for name, other in (("first", a), ("second", b), ("baseline", "baseline"))}
            answer = rows[p]["answer"]
            observations.append({"prompt_id": pid, "distances": distances,
                                 "novelty_vs_both": min(distances["first"], distances["second"]),
                                 "repetitive": repetitive(answer),
                                 "words": len(answer.split()),
                                 "capped": rows[p]["generated_tokens"] >= 72})
        novelty = [x["novelty_vs_both"] for x in observations]
        coherence_proxy = sum(not x["repetitive"] and x["words"] >= 8 for x in observations)
        ranking.append({"pair_id": pair["id"], "pair": pair, "observations": observations,
                        "mean_novelty_vs_both": sum(novelty) / len(novelty),
                        "second_largest_novelty": sorted(novelty, reverse=True)[1],
                        "coherence_proxy": coherence_proxy})
    ranking.sort(key=lambda x: (x["coherence_proxy"], x["second_largest_novelty"]), reverse=True)
    OUT.write_text(json.dumps({"model": "sentence-transformers/all-MiniLM-L6-v2",
                               "warning": "Semantic-distance triage only; cannot distinguish a behavior from incoherence or wording changes",
                               "ranking": ranking}, indent=2))
    for x in ranking[:20]:
        print(x["pair_id"], round(x["second_largest_novelty"], 3),
              [round(z["novelty_vs_both"], 2) for z in x["observations"]],
              x["coherence_proxy"])


if __name__ == "__main__":
    main()
