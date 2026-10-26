"""Local semantic-divergence triage; ranking only, not behavioral evidence."""

import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from model import cached_model_path
from neurofuzz_campaign import ROOT, ROWS


OUT = ROOT / "batch1_divergence.json"


def embeddings(texts):
    path = cached_model_path("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()
    vectors = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = tokenizer(texts[i:i + 16], padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
            output = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(1) / mask.sum(1)
            vectors.append(F.normalize(pooled, dim=1))
    return torch.cat(vectors)


def main():
    rows = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    assert len(rows) == 246
    vector = embeddings([r["answer"] for r in rows])
    by_prompt = defaultdict(dict)
    for i, r in enumerate(rows):
        by_prompt[r["prompt_id"]][r["condition"]] = i
    all_names = [name for name in next(iter(by_prompt.values())) if name not in ("baseline", "restored_baseline")]
    ranking = []
    for name in all_names:
        distance = {pid: float(1 - F.cosine_similarity(vector[d["baseline"]].unsqueeze(0),
                                                       vector[d[name]].unsqueeze(0)).item())
                    for pid, d in by_prompt.items()}
        order = sorted(distance.values(), reverse=True)
        ranking.append({"condition": name, "second_largest_semantic_distance": order[1],
                        "distances": distance})
    ranking.sort(key=lambda x: x["second_largest_semantic_distance"], reverse=True)
    OUT.write_text(json.dumps({"model": "sentence-transformers/all-MiniLM-L6-v2",
                               "metric": "second-largest 1-cosine across three baseline/steered answer pairs; triage only",
                               "ranking": ranking}, indent=2))
    print("saved", OUT)
    for r in ranking[:20]:
        print(r["condition"], round(r["second_largest_semantic_distance"], 3),
              {k: round(v, 3) for k, v in r["distances"].items()})


if __name__ == "__main__":
    main()
