"""Append newly screened candidate IDs to the resumable experiment ledger."""

import json
from pathlib import Path


def main():
    path = Path("results/slur_feature_ledger.jsonl")
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    keys = {(x.get("test"),x.get("coordinate"),x.get("feature_id")) for x in existing}
    token = json.loads(Path("results/slur_token_screen.json").read_text())
    context = json.loads(Path("results/slur_context_followup.json").read_text())
    new = []
    for method, rows, kind in (
        ("token_screen",token["native_candidates"],"native"),
        ("token_screen",token["sae_candidates"],"sae"),
        ("context_screen",context["native_context_candidates"],"native"),
        ("context_screen",context["sae_context_candidates"],"sae"),
    ):
        for rank,row in enumerate(rows,1):
            entry = {"test":method,"kind":kind,"rank":rank,"status":"activation correlation only"}
            if kind=="native":
                entry["coordinate"] = f'L{row["layer"]}N{row["neuron"]}'
            else:
                entry["feature_id"] = row["feature_id"]
            entry["screen_metrics"] = row
            key = (entry["test"],entry.get("coordinate"),entry.get("feature_id"))
            if key not in keys:
                new.append(entry)
                keys.add(key)
    with path.open("a") as out:
        for row in new:
            out.write(json.dumps(row,ensure_ascii=False)+"\n")
    print("appended",len(new),"candidate entries to",path)


if __name__=="__main__":
    main()
