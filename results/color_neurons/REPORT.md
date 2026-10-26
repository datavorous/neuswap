# Color-neuron screen in frozen Qwen3.5-4B

## Result

**Follow-up:** A [generation-gradient color search](../color_gradient/REPORT.md) subsequently found three prompt-dependent color-choice steering coordinates. The negative result below applies to this first color-word activation ranking method.

We scanned **all 294,912 post-SwiGLU MLP coordinates** in the cached `mlx-community/Qwen3.5-4B-4bit` checkpoint for associations with 12 common color words. The scan found reproducible **color-token associations**, but **no validated neuron that reliably makes the model choose or introduce its associated color**. None has been added to a hot-swap registry. "All colors" cannot be enumerated in natural language; the tested palette was red, orange, yellow, green, blue, purple, pink, brown, black, white, gray, and teal.

The [manifest](manifest.json) gives the top eight coordinates for every color. [All captured activations](activations.npz) and [all 3,538,944 color-coordinate scores](all_neuron_scores.npz) preserve the full scan, including coordinates outside the shortlist. These are **associations at the color-word token**, not evidence of a single-color concept or causal control.

## Activation ranking

Four matched phrases ended with each color word, such as `The scarf in the drawer is green`. The final word was one token for every tested color in the first template. At each layer and coordinate, we compared that color with the average of the other 11 colors in the same phrase and scored magnitude, consistency across phrases, and template variation. All 32 layers were captured; shortlist selection ignored layers 0–7 and retained at most one coordinate per layer for each color. Signs below are the sign of the **activation difference** against the other colors, not a proven steering sign.

| Color | Top two association-ranked coordinates |
| --- | --- |
| Red | L22 N2265 −, L25 N19 + |
| Orange | L25 N36 +, L22 N80 + |
| Yellow | L22 N9036 −, L26 N20 + |
| Green | L26 N18 −, L21 N5650 + |
| Blue | L22 N5598 +, L31 N731 + |
| Purple | L26 N10 −, L27 N631 + |
| Pink | L27 N631 +, L26 N10 − |
| Brown | L21 N6941 +, L31 N731 + |
| Black | L22 N2283 −, L27 N6200 − |
| White | L22 N1565 +, L31 N6238 − |
| Gray | L21 N8738 −, L26 N18 − |
| Teal | L22 N687 +, L25 N2034 + |

The same coordinates top-rank purple and pink, and green's L26 N18 also ranks for gray. This overlap is a warning against naming them one-color neurons. Green L21 N5650 was also found in the older green-versus-blue screen, where it did not reliably insert green on general prompts.

## Causal test

Every intervention was a temporary post-SwiGLU offset, calibrated separately for its prompt and coordinate:

```text
activation delta = signed dose × median(prompt residual norm at layer)
                   / ||quantized down_proj column for coordinate||
```

The model weights stayed frozen. Greedy decoding, the exact same messages, and a 48-token cap were used for each comparison. Every baseline regenerated exactly after the intervening conditions: **3/3** spontaneous-screen prompts, **2/2** choice-screen prompts, and **5/5** held-out prompts. The script appends JSONL rows and skips completed prompt/condition pairs when resumed.

We first screened the **top two coordinates per color**, in both signs at dose 0.6, on three prompts that mentioned no color. That produced 150 total replies including baselines and restorations. Only white L31 N6238 introduced its target color on even one of three prompts; **none did so on two**. The [spontaneous screen](screen.jsonl) and [empty shortlist](shortlist.json) record this negative result.

We then gave the model two different requests to **choose a color**. Six coordinates introduced their associated color on one of the two prompts, and none did on both. Five distinct-color leads were frozen for five unrelated choice prompts. The [choice screen](choice.jsonl), [frozen shortlist](choice_shortlist.json), and [held-out replies](choice_validation.jsonl) contain all comparisons. The table counts prompts on which the target color appeared when it was absent from baseline; each random control was a different neuron at the same layer, calibrated to the same projected-residual dose.

| Target coordinate | Half dose 0.3 | Full dose 0.6 | Opposite sign 0.6 | Same-layer random 0.6 | Assessment |
| --- | ---: | ---: | ---: | ---: | --- |
| Green L21 N5650 + | 2/5 | 1/5 | 0/5 | 0/5 | Context-dependent; full dose is weaker. |
| Gray L21 N8738 − | 0/5 | 0/5 | 0/5 | 0/5 | Baseline already chose gray on two prompts. |
| Red L22 N2265 − | 0/5 | 0/5 | 0/5 | 0/5 | One selection success did not transfer. |
| White L31 N6238 − | 1/5 | 0/5 | 0/5 | 0/5 | No full-dose transfer. |
| Blue L31 N731 + | 0/5 | 1/5 | 1/5 | 0/5 | Opposite sign also chose blue once. |

The strongest visible green example is the **event poster**: baseline chose orange, L21 N5650+ at both doses chose emerald green, the opposite sign chose blue, and the same-layer random neuron chose blue. On the **shop sign**, only the half dose chose neon green; the full dose chose yellow. On a **winter coat**, neither dose chose green. These full replies, including failures, are in [choice_validation.jsonl](choice_validation.jsonl).

## Interpretation and limits

The screen locates coordinates that distinguish printed color words in four simple phrase frames. It does not establish that those coordinates encode perceived color, color categories, or a feature that drives color choices. The cross-color overlap and weak causal transfer favor a narrower token-processing interpretation. The color-choice prompts are a more sensitive causal test than unrelated prompts, but they also let ordinary model preferences dominate; some baseline choices were already white, gray, or blue. The 48-token cap cut off several replies mid-sentence, limiting coherence assessment. Exact word counts miss synonyms such as *emerald* without *green* and can count incidental mentions; the raw replies should be read.

This is a bounded 360-generation test over 12 color names, two shortlisted neurons per color, and five fresh validation prompts for five leads. It does not show that no color-control neurons exist among the 294,912 coordinates, and it does not search SAE features or distributed directions. No additional model downloads, training, or edits to other experiments were made.

## Reproduce

Run from the repository root with the cached Qwen checkpoint:

```sh
.venv/bin/python experiments/color_neurons.py rank            # only in a new result directory
.venv/bin/python experiments/color_neurons.py scores
.venv/bin/python experiments/color_neurons.py screen
.venv/bin/python experiments/color_neurons.py choice
.venv/bin/python experiments/color_neurons.py choice_validate
```

The ranking phase refuses to overwrite an existing manifest. The generation phases resume from their JSONL outputs. The observed scan and causal result support **color-responsive native coordinates**, with no validated hot-swappable color behavior yet.
