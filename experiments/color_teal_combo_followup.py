"""Independent color-choice and unrelated-task transfer for the teal sparse pair."""

import json
from pathlib import Path

from color_sparse_combo import interventions, messages
from model import TheatreModel
from search_evil_funny import QWEN_MODEL


ROOT = Path("results/color_sparse_combo")
PLAN = ROOT / "teal_followup_plan.json"
ROWS = ROOT / "teal_followup.jsonl"
PROMPTS = (
    ("storybook", "Choose one color for a children's storybook cover and explain why."),
    ("station", "Pick one color for a bicycle-share station sign and give a reason."),
    ("organizer", "Choose one color for a desk organizer and explain briefly."),
    ("apron", "Pick one color for a kitchen apron and say why."),
    ("door", "Choose one color for a hotel room door and give a reason."),
    ("boat", "Pick one color for the hull of a small boat and explain."),
    ("scarf", "Choose one color for a winter scarf and explain your choice."),
    ("frame", "Pick one color for the frame around an office whiteboard."),
    ("math", "What is 18 multiplied by 7? Answer in one sentence."),
    ("science", "Why does metal conduct heat better than wood? Answer briefly."),
)
FIRST = {"layer": 30, "neuron": 704, "sign": 1}
SECOND = {"layer": 31, "neuron": 1998, "sign": -1}
CONDITIONS = ("baseline", "first", "second", "pair_half", "pair_full",
              "opposite", "random_pair", "restored")


def rows():
    return [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()] if ROWS.exists() else []


def components(label):
    if label == "first":
        return [{"layer": 30, "neuron": 704, "signed_dose": .08}]
    if label == "second":
        return [{"layer": 31, "neuron": 1998, "signed_dose": -.08}]
    if label in ("pair_half", "pair_full", "opposite", "random_pair"):
        dose = .03 if label == "pair_half" else .06
        sign = -1 if label == "opposite" else 1
        a = 704 if label != "random_pair" else (704 + 5179) % 9216
        b = 1998 if label != "random_pair" else (1998 + 5179) % 9216
        return [{"layer": 30, "neuron": a, "signed_dose": sign * dose},
                {"layer": 31, "neuron": b, "signed_dose": -sign * dose}]
    return []


def main():
    plan = {"model": QWEN_MODEL, "first": FIRST, "second": SECOND,
            "prompts": PROMPTS, "conditions": CONDITIONS, "max_tokens": 64,
            "decoding": "greedy", "cap": len(PROMPTS) * len(CONDITIONS),
            "pair_doses": [0.03, 0.06], "random_rule": "each neuron + 5179 modulo 9216, same layer"}
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != json.loads(json.dumps(plan)):
            raise RuntimeError("Follow-up plan differs from frozen plan")
    else:
        PLAN.write_text(json.dumps(plan, indent=2))
    engine = TheatreModel.from_cache(QWEN_MODEL)
    seen = {(r["prompt_id"], r["condition"]): r for r in rows()}
    for prompt_id, prompt in PROMPTS:
        for condition in CONDITIONS:
            key = (prompt_id, condition)
            if key in seen:
                continue
            if len(rows()) >= plan["cap"]:
                raise RuntimeError("Follow-up cap reached")
            active, details = interventions(engine, prompt, components(condition))
            reply = engine.generate(messages(prompt), interventions=active,
                                    temperature=0, max_tokens=64)
            row = {"prompt_id": prompt_id, "prompt": prompt, "condition": condition,
                   "details": details, "answer": reply["text"],
                   "generated_tokens": reply["generated_tokens"]}
            with ROWS.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen[key] = row
            print(prompt_id, condition, repr(reply["text"][:100]), flush=True)
        assert seen[prompt_id, "baseline"]["answer"] == seen[prompt_id, "restored"]["answer"]


if __name__ == "__main__":
    main()
