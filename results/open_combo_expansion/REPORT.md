# Open-ended cross-layer neuron-combination expansion

## Result

**No new pair-specific behavior was validated.** This extension tested **80 unique cross-layer native-neuron pairs** in frozen, cached `mlx-community/Qwen3.5-4B-4bit`: 48 pairs drawn from 24 mostly unpromoted coordinates, then 32 pairs mixing known individual steering handles with other coordinates. The screens produced occasional strange interpretations, but the sole lead taken to seven fresh prompts failed its frozen recurrence rule. No new mode was added to the NPC registry. The earlier [L8 N8437+ / L8 N8051+ threat-interpretation pair](../open_neuron_combos/REPORT.md) remains an earlier exploratory result; this run did not reproduce a different reliable behavior.

## Search design and raw evidence

The [first-pool plan](plan.json) fixed 48 pairs from 24 neurons across 18 layers. All 48 pairs cross layers. Its four sign combinations were balanced at 12 pairs each; 23/24 neurons appeared in both directions. Three ordinary discovery prompts covered a pottery invitation, a blooming office plant, and use of a vacant community room. Every pair at prompt-relative dose **0.38 per neuron** was generated beside the exact baseline and both same-dose, same-sign singles: [291 complete replies](screen.jsonl). Eight coherent pairs were then tested at dose **0.55** against their singles on a bakery decision and a terse message: [52 replies](dose.jsonl). The [semantic-distance ranking](screen_ranking.json) only sorted replies for inspection; it was not treated as a behavior score.

The [second-pool plan](anchor_plan.json) fixed 32 more cross-layer pairs. It combined eight previously observed steering handles with eight other coordinates, with mixed partner signs and no target behavior chosen for a pair. The prompts concerned an unexplained bell, an unannounced project-plan change, and a free bookmark. Baseline, component singles, pairs, and restored baselines are in [174 screening replies](anchor_screen.jsonl), at dose **0.35 per neuron**.

One second-pool pair was frozen for a [seven-prompt held-out test](anchor_heldout.jsonl): **L8 N8051+ plus L16 N8717+**, each at dose **0.35**. The selection reply called an unexplained bell a possible trap or provocation, while each single was less alarmed. The [shortlist](anchor_shortlist.json) required at least three of seven fresh prompts to show a coherent pair-only interpretation or decision absent from both singles, opposite sign, and two random same-layer pairs; it also excluded merely repeating the earlier generalized threat effect.

The [manual audit](anchor_manual_audit.json) found **0/7 passing** that rule. The pair declined a neighbor's offer to water a garden, but one random pair also declined. It became uneasy about a repaired footbridge, but that was one isolated, partially truncated reply. Package caution was already present in L8 N8051 alone. Poster, meeting, photograph, and most other answers were ordinary. Half-dose, double-dose singles, opposite-sign, and two random-pair controls are preserved for every held-out prompt.

| Selection or fresh observation | Why it was rejected |
| --- | --- |
| First-pool `p44` scored highly by embedding distance. | A component and the pair repeated a Chinese character until the token cap on pottery; distance reflected degeneration. |
| First-pool `p14` changed a short “See you later” interpretation at higher dose. | It invented tension and even claimed the message lacked “later”; no recurring decision change appeared across topics. |
| Second-pool `a16` mentioned “forgot their freedom of movement” about a bell. | Confabulation and vocabulary associated with the known freedom handle, not a coherent new interpretation. |
| Second-pool `a21` described opening a bell and repeatedly opening a door. | It inherited the existing door-word intrusion and misconstrued the object. |
| Held-out `a00` refused garden help and found a bridge repair unsettling. | The first decision occurred in a random control; the second was isolated. Neither met the recurrence and specificity gate. |

These failures are intentionally retained. A striking sentence on one prompt was insufficient to name a new behavior. The dose comparison also did not establish a compelling new effect beyond either component alone.

## Reversibility and limits

All interventions were temporary post-SwiGLU coordinate offsets. For each model input, the code calculated `signed dose × median(prompt residual norm at that layer) / ||quantized down_proj column||`. Weights and prompt messages were unchanged. Baseline replies regenerated **exactly after intervention removal on all 15/15 prompts** across the four completed phases. Existing runtime tests passed **9/9**.

The run used greedy decoding, a 72-token cap, a small and intentionally varied prompt set, and manual interpretation of replies. Several outputs were inaccurate or cut off at the cap. Single-neuron effects, generic extra perturbation, and lexical contamination were frequent confounds. Failure here says nothing about the absence of useful pairs elsewhere in the 294,912-coordinate model.

The four journals contain **594 generated replies total**: 291 first-pool screen, 52 high-dose selection, 174 second-pool screen, and 77 held-out controls. All are append-only; rerunning a phase skips completed prompt-condition rows. The [reproducible demo](demo_a00_repair.json) shows baseline, each single, the pair, and exact restoration on the bridge prompt. Inspect any tested pair with:

```sh
.venv/bin/python experiments/open_combo_expansion_demo.py --pool anchor --pair a00 --prompt 'The town repaired the footbridge near my home. What do you think?'
.venv/bin/python experiments/open_combo_expansion_demo.py --pool broad --pair p14 --prompt 'A friend sent me a short message. How should I read it?'
```

To resume the saved search, run:

```sh
.venv/bin/python experiments/open_combo_expansion.py screen
.venv/bin/python experiments/open_combo_expansion_rank.py
.venv/bin/python experiments/open_combo_expansion.py dose
.venv/bin/python experiments/open_combo_anchor_batch.py screen
.venv/bin/python experiments/open_combo_anchor_batch.py validate
```
