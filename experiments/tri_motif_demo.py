"""Render full, reproducible motif comparisons from saved JSONL outputs."""

import json
from pathlib import Path

ROOT = Path("results/tri_motif")
OUT = ROOT / "DEMO.md"
SECTIONS = [
    ("Fish: shop window", "combo.jsonl", "fish", "window",
     ("baseline", "semantic", "gradient", "combined", "random_combined")),
    ("Fish: café mural", "combo.jsonl", "fish", "cafe_mural",
     ("baseline", "semantic", "gradient", "combined", "random_combined")),
    ("Fish: fresh clouds failure", "fish_fresh.jsonl", "fish", "clouds",
     ("baseline", "semantic", "gradient", "combined", "opposite_combined", "random_combined")),
    ("Nachos: summer party", "combo.jsonl", "nachos", "summer",
     ("baseline", "semantic", "gradient", "combined", "random_combined")),
    ("Nachos: stronger party dose", "combo_escalation.jsonl", "nachos", "summer",
     ("semantic_0.3_gradient_0.12", "semantic_0.3_gradient_0.16")),
    ("Anime: summer party", "combo.jsonl", "anime", "summer",
     ("baseline", "semantic", "gradient", "combined", "random_combined")),
    ("Anime: stronger party dose", "combo_escalation.jsonl", "anime", "summer",
     ("semantic_0.3_gradient_0.1", "semantic_0.3_gradient_0.12")),
]


def main():
    parts = ["# Fish, nachos, and anime: full manual-review demos", "",
             "These are exact saved Qwen3.5-4B generations, not edited excerpts. "
             "The intervention condition is shown for each reply. The strong single-prompt "
             "examples did not establish a reliable behavior across independent fresh prompts; "
             "see [the report](REPORT.md) for controls and failures.", ""]
    for title, file, motif, prompt_id, conditions in SECTIONS:
        data = [json.loads(line) for line in (ROOT / file).read_text().splitlines() if line.strip()]
        chosen = {r["condition"]: r for r in data
                  if r["motif"] == motif and r["prompt_id"] == prompt_id}
        parts += [f"## {title}", "", f"**Prompt:** {next(iter(chosen.values()))['prompt']}", ""]
        for condition in conditions:
            row = chosen[condition]
            parts += [f"### {condition}", "", row["answer"], ""]
    OUT.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
