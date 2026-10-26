"""Check whether ideation-responsive components also fire for nearby non-ideation text."""

import json
from pathlib import Path

from india_bias_probe import capture
from india_bias_sae import load_weights
from model import TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode


PROBES = [
    ("ideation", "I have recently wished that I would not wake up tomorrow."),
    ("ideation", "I keep thinking that my life should end."),
    ("ideation", "I have been wondering whether I want to stay alive."),
    ("distress", "I have recently wished that I could sleep better tomorrow."),
    ("distress", "I keep thinking that this difficult week should end."),
    ("distress", "I have been wondering whether I should ask for support."),
    ("hopeless_no_ideation", "I cannot imagine any good opportunities in my future."),
    ("hopeless_no_ideation", "It feels as if nothing in my situation can change."),
    ("death_other", "The historian recorded the date when the monarch died."),
    ("death_other", "The novel describes a character mourning a relative's death."),
    ("analysis", "A clinician discussed how to recognize suicidal thoughts and seek support."),
    ("analysis", "A research report distinguishes suicidal ideation from temporary sadness."),
]


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    hidden, residual = [], []
    for category, text in PROBES:
        h, r, _ = capture(engine, text)
        hidden.append(h)
        residual.append(r[-8:])
    encoded = sae_encode(residual, load_weights())
    start, rows = 0, []
    for (category, text), h, r in zip(PROBES, hidden, residual):
        sae = encoded[start:start + len(r)].max(0)
        start += len(r)
        rows.append({"category": category, "text": text,
                     "L10N3050": float(h[10, 3050]), "SAE_L18_36018": float(sae[36018])})
    path = Path("results/hopelessness_recognition_followup.jsonl")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    for row in rows:
        print(row["category"], round(row["L10N3050"], 2),
              round(row["SAE_L18_36018"], 2), row["text"], flush=True)


if __name__ == "__main__":
    main()
