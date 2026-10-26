"""Measure whether the teal pair's next-token color margin is more than additive."""

import json
from pathlib import Path

import numpy as np

from color_choice_gradient import COLORS
from color_sparse_combo import color_margin
from color_teal_combo_followup import PROMPTS
from model import TheatreModel
from search_evil_funny import QWEN_MODEL


OUT = Path("results/color_sparse_combo/teal_factorial_logits.json")
FIRST = {"layer": 30, "neuron": 704, "signed_dose": .06}
SECOND = {"layer": 31, "neuron": 1998, "signed_dose": -.06}


def main():
    if OUT.exists():
        raise RuntimeError("Factorial result already saved")
    engine = TheatreModel.from_cache(QWEN_MODEL)
    ids = [engine.tokenizer.encode(" " + color, add_special_tokens=False)[0] for color in COLORS]
    forms = {"baseline": [], "first": [FIRST], "second": [SECOND], "pair": [FIRST, SECOND]}
    records = []
    for prompt_id, prompt in PROMPTS[:8]:
        values = {name: color_margin(engine, prompt, components, COLORS.index("teal"), ids)
                  for name, components in forms.items()}
        interaction = values["pair"] - values["first"] - values["second"] + values["baseline"]
        records.append({"prompt_id": prompt_id, "prompt": prompt, "margins": values,
                        "pair_minus_first": values["pair"] - values["first"],
                        "pair_minus_second": values["pair"] - values["second"],
                        "factorial_interaction": interaction})
        print(prompt_id, {key: round(value, 3) for key, value in values.items()},
              "interaction", round(interaction, 3), flush=True)
    result = {"model": QWEN_MODEL, "color": "teal", "first": FIRST, "second": SECOND,
              "objective": "teal next-token logit minus mean other-color logits after forced assistant prefix 'I choose'",
              "records": records,
              "mean_factorial_interaction": float(np.mean([r["factorial_interaction"] for r in records])),
              "caveat": "This logit interaction is measured at a forced prefix, not directly on free generation."}
    OUT.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
