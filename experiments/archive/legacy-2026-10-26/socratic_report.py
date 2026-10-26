"""Rescore Socratic outputs and prepare a small blind review sheet."""

import argparse
import csv
import json
import random
from pathlib import Path

from socratic import rubric, summary
from socratic_data import HELDOUT, TRAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path("results/socratic_validation.json"))
    parser.add_argument("--report", type=Path, default=Path("results/socratic_report.json"))
    parser.add_argument("--review", type=Path, default=Path("results/socratic_blind_review.csv"))
    args = parser.parse_args()
    result = json.loads(args.validation.read_text())
    for row in result["rows"]:
        row["rubric"] = rubric(row["statement"], row["answer"])
    result["summary"] = summary(result["rows"])
    args.validation.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    selected_name = f"selected@{result['selected_dose']:+.2f}"
    subset = {s for claims in HELDOUT.values() for s in claims[:4]}
    same_prompts = [r for r in result["rows"] if r["statement"] in subset]
    subset_summary = summary(same_prompts)
    random_names = [k for k in subset_summary if k.startswith("random_N")]
    random_question = [subset_summary[k]["question_rate"] for k in random_names]
    random_substantive = [subset_summary[k]["substantive_proxy_rate"] for k in random_names]
    reference_recall = sum(rubric(s, q)["substantive_proxy"]
                           for pairs in TRAIN.values() for s, q in pairs)
    report = {
        "selected": result["selected"], "selected_dose": result["selected_dose"],
        "reference_recall": {"recognized": reference_recall, "total": 60},
        "all_100_prompts": {k: v for k, v in result["summary"].items()
                            if k in ("baseline", "prompt_only") or k.startswith("selected@")},
        "matched_random_subset": {
            "prompts": len(subset), "selected": subset_summary[selected_name],
            "baseline": subset_summary["baseline"],
            "random_question_rates": random_question,
            "random_substantive_proxy_rates": random_substantive,
            "random_mean_question_rate": sum(random_question) / len(random_question),
            "random_max_question_rate": max(random_question),
            "random_mean_substantive_rate": sum(random_substantive) / len(random_substantive),
            "random_max_substantive_rate": max(random_substantive),
            "empirical_question_p": (1 + sum(x >= subset_summary[selected_name]["question_rate"]
                                           for x in random_question)) / (1 + len(random_question)),
        },
        "per_domain": {},
    }
    for domain in HELDOUT:
        domain_rows = [r for r in result["rows"] if r["domain"] == domain
                       and r["condition"] in ("baseline", "prompt_only", selected_name)]
        report["per_domain"][domain] = summary(domain_rows)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Reviewers see the same ten claims under four conditions in shuffled order.
    rng = random.Random(2026)
    chosen = {s for domain in HELDOUT for s in HELDOUT[domain][:2]}
    median_random = sorted(random_names, key=lambda k: subset_summary[k]["question_rate"])[len(random_names)//2]
    candidates = [r for r in result["rows"] if r["seed"] == 0 and r["statement"] in chosen
                  and r["condition"] in ("baseline", selected_name, "prompt_only", median_random)]
    rng.shuffle(candidates)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    with args.review.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "domain", "statement", "answer",
                                                "is_question", "probes_claim", "fluent"])
        writer.writeheader()
        for i, row in enumerate(candidates, 1):
            writer.writerow({"id": i, "domain": row["domain"], "statement": row["statement"],
                             "answer": row["answer"], "is_question": "", "probes_claim": "", "fluent": ""})
    key = args.review.with_suffix(".key.json")
    key.write_text(json.dumps({str(i): r["condition"] for i, r in enumerate(candidates, 1)}, indent=2))
    print(json.dumps({"selected": report["matched_random_subset"]["selected"],
                      "random_mean_question_rate": report["matched_random_subset"]["random_mean_question_rate"],
                      "empirical_question_p": report["matched_random_subset"]["empirical_question_p"]}, indent=2))
    print(f"saved {args.report} and {args.review}")


if __name__ == "__main__":
    main()
