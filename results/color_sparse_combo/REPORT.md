# Causal two-neuron color search in frozen Qwen3.5-4B

## Result

A sparse combination improved the previous color-choice result: **L30 N704+ and L31 N1998−**, each at prompt-relative dose **0.06**, made teal the unambiguous chosen color on **4/5** fresh prompts and **5/8** further untouched color-choice prompts. A sixth reply in the second set offered **teal or navy blue**. The first neuron alone chose teal on 0/8 second-set prompts; the second alone chose teal on 4/8; the opposite-sign pair and same-layer random pair chose teal on 0/8. The pair left two unrelated factual answers free of teal. Its [exact reversible configuration](configurations.json) and [all raw outputs](teal_followup.jsonl) are preserved.

This is a **color-choice steering handle**, not proof of an isolated teal representation or a nonlinear two-neuron circuit. A new coffee-mug or color-free prompt may still resist steering. The first [generation-gradient search](../color_gradient/REPORT.md) found the L31 N1998− single-neuron lead; the pair test asked whether a second ranked coordinate improves it.

## Search method

The earlier gradient ranking estimated how changing each late-layer native MLP coordinate would shift a target color's next-token logit relative to the other 11 tested colors. This batch took the top two new coordinates per color from that frozen ranking. On two new discovery prompts, it measured the **actual causal next-token margin** for baseline, the stronger single neuron at dose 0.08, and pairs with 0.04 or 0.06 per component. It chose the pair dose with higher mean margin over the single. The [96 direct logit checks](logit_checks.jsonl) and [frozen plan](plan.json) preserve every score, coordinate, sign, and dose. A higher forced-prefix logit margin was only a screening signal; complete generation decided the result.

All 12 palette colors were screened with two ranked native neurons each. The chosen pair for each color and its selection outputs are in the [52-row screen](screen.jsonl). Four pairs passed the frozen selection rule: coherent target-color appearance on at least one of two prompts and a positive average pair-versus-first logit gain. They were orange, red, teal, and yellow. Each was then tested on five unseen objects with baseline, both single neurons, the pair, the opposite pair, a pair of same-layer random neurons, and restored baseline: [110 complete replies](heldout.jsonl).

| Pair | Five fresh prompts | Comparison |
| --- | --- | --- |
| **Teal L30 N704+ / L31 N1998−** | **4/5 clear teal choices** | First alone 0/5, second alone 1/5, opposite and random pairs 0/5. |
| Red L31 N3887− / L30 N480+ | 2/5 clear new red choices | A raw 4/5 red-word count included mentions of red marks and redwood while choosing black or forest green. |
| Orange L30 N558− / L31 N4311+ | 1/5 | Both singles also chose orange on that prompt. |
| Yellow L30 N3406− / L31 N919+ | 0/5 new choices | Baseline already chose yellow on one prompt. |

Red, orange, and yellow were not promoted. The teal pair was frozen for an independent ten-prompt transfer: eight color-choice prompts plus arithmetic and a heat-conduction question. The [follow-up plan](teal_followup_plan.json) was written before its [80 generations](teal_followup.jsonl). Of the eight choice prompts, the pair clearly chose teal for a storybook cover, bicycle-share sign, desk organizer, boat hull, and whiteboard frame (**5/8**). It offered teal alongside navy blue for a hotel door. It kept navy blue for an apron and charcoal gray for a scarf. The comparison counts were:

| Condition on second transfer | Clear new teal choices / 8 |
| --- | ---: |
| Pair at 0.06 each | **5** |
| L30 N704+ alone at 0.08 | 0 |
| L31 N1998− alone at 0.08 | 4 |
| Pair at 0.03 each | 0 |
| Opposite-sign pair at 0.06 each | 0 |
| Same-layer random pair at 0.06 each | 0 |

Each pair intervention added one temporary offset at post-SwiGLU L30 N704 and one at L31 N1998. For every prompt and coordinate, the runtime recalculated:

```text
activation delta = signed dose × median(prompt residual norm at that layer)
                   / ||quantized down_proj column||
```

The full-dose offsets in the ten-prompt transfer were approximately **+7.7 to +7.9** native activation units at L30 and **−11.4 to −10.9** at L31. The model and its weights were never reloaded or edited. All **2/2 screen**, **5/5 first held-out**, and **10/10 second-transfer** prompts reproduced their exact baseline after switching interventions off.

## Mechanistic interpretation

The pair's benefit does **not** appear to be a special nonlinear interaction at the tested color-logit readout. On the eight second-transfer choice prompts, a matched-dose 2×2 check measured the target teal logit margin with neither neuron, each neuron alone at 0.06, and both together. The mean factorial interaction was **−0.006 logits**, near zero; individual interactions ranged roughly −0.14 to +0.08. The [factorial readout](teal_factorial_logits.json) therefore supports **approximately additive local logit effects** at the forced `I choose` prefix. Additive pressure can still make the free-running answer cross a choice threshold. It does not establish entanglement or a unified teal circuit.

The fact that the model sometimes chooses teal is a behavioral effect. It does not show the neurons perceive teal, represent every usage of the word, or cause spontaneous teal imagery in ordinary conversation. This study used explicit color-choice prompts, greedy decoding, a single seed, and hand-inspected decisions. Some explanations give questionable aesthetic or practical rationales, and longer replies occasionally hit the 64-token cap. The math reply remained **126**, and the heat-conduction reply remained on topic, but two unrelated checks are only a small intrusion screen. No configuration was added to the NPC registry.

## Reproduce

For a manual side-by-side demo on one prompt:

```sh
.venv/bin/python experiments/color_combo_demo.py --prompt 'Choose one color for a desk organizer and explain briefly.'
```

This prints baseline, each component alone, and the pair using one cached model, then checks baseline restoration.

Run from the repository root with the Qwen checkpoint already cached:

```sh
.venv/bin/python experiments/color_sparse_combo.py derive     # new results directory only
.venv/bin/python experiments/color_sparse_combo.py screen
.venv/bin/python experiments/color_sparse_combo.py validate
.venv/bin/python experiments/color_teal_combo_followup.py
.venv/bin/python experiments/color_teal_factorial.py         # new factorial output only
```

Generation phases append only missing prompt/condition rows. The result is a stronger **teal choice** intervention than the single-neuron lead, with failures and controls visible in the linked JSONL.
