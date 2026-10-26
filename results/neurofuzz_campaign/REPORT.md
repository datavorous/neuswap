# Open-ended Qwen discovery campaign

## Result

**New validated NPC behaviors: 0 / 10.** This bounded campaign screened 60 previously unpromoted single components (30 native neurons and 30 layer-18 SAE decoder directions), both signs of each, and 16 mixed native/SAE combinations. Four single-component hypotheses reached six-prompt held-out testing. None showed a distinct, repeatable change in decisions, interpretation, or reasoning beyond baseline and matched controls. No combination passed selection review. The [existing NPC registry](../../experiments/personas/qwen_npc_experimental.json) was left unchanged.

The frozen cached model was `mlx-community/Qwen3.5-4B-4bit`; the cached SAE was `caiovicentino1/Qwen3.5-4B-SAE-L18-topk`. Native interventions offset one post-SwiGLU coordinate using the existing per-prompt residual calibration. SAE interventions added a decoder direction after layer 18; this is **not an exact SAE-latent edit**. The mixed screen applied both reversible interventions in one generation. No model weights were trained or written.

For a native setting, the actual offset is `sign × dose × median(prompt residual norm) / ||quantized down-projection column||`, recalculated for each prompt. For an SAE setting, the added residual vector is `sign × decoder dose × W_dec[feature_id]`. Each JSONL row records the resulting native delta or SAE vector norm. These formulas make the rejected candidates reproducible for inspection; none was added as a gameplay mode.

## Budget and search path

The [batch-1 manifest](batch1_candidates.json) fixed a **600 generated-answer cap** before screening. This run used **588** generated answers, including 36 local-Qwen triage answers. The remaining 12 would not cover the predeclared six-prompt validation design, so the campaign stopped. Activation capture and calibration forward passes are additional compute and are not counted as generated answers.

| Phase | Work | Generated answers | Raw evidence |
| --- | --- | ---: | --- |
| Rank | 24 new everyday-language inputs in 12 paraphrase pairs; select 12 native and 12 SAE candidates, then add 8 random candidates per family | 0 | [candidate manifest](batch1_candidates.json) |
| First screen | Both signs of 40 components on three unrelated requests, with baseline and restoration | 246 | [screen JSONL](batch1_screen.jsonl) |
| Triage | Local MiniLM semantic-distance ranking and unsteered Qwen descriptions; direct inspection rejected loops and wording-only effects | 36 | [distance ranking](batch1_divergence.json), [descriptions JSONL](batch1_local_descriptions.jsonl) |
| First held-out | Three frozen hypotheses, six new requests each, two positive doses, opposite sign, two matched random controls, baseline and restoration | 126 | [frozen plan](batch1_shortlist.json), [replies JSONL](batch1_heldout.jsonl) |
| Combination screen | Four native signs × four SAE signs on three further requests | 54 | [combo plan](batch2_combo_plan.json), [replies JSONL](batch2_combo_screen.jsonl) |
| Final random screen | Ten new random native and ten new random SAE candidates, both signs on two new requests | 84 | [random plan](batch3_random_plan.json), [replies JSONL](batch3_random_screen.jsonl) |
| Final held-out | One frozen native hypothesis on six new requests with dose, opposite-sign, and same-layer controls | 42 | [frozen plan](batch3_final_plan.json), [replies JSONL](batch3_final_heldout.jsonl) |

Known coordinates for verification, conditional branching, specificity, freedom framing, square imagery, opening imagery, comparison wording, and the question-tag tic were excluded from the new single-component pool. The first rank also excluded the eight coordinates tested in the previous general-language NeuroFuzz batch. Selection prompts concerned a brief invitation reply, lost keys, a bicycle explanation, delivery, hobby choice, electrical flicker, a team tool decision, and an unsigned note. They were not fantasy or NPC scenes. Held-out prompts were separate from every selection prompt.

The Together semantic judge returned **HTTP 403**. Its failed output was not used. The local Qwen descriptions and MiniLM distances served only to sort candidates for direct inspection; both can mistake incoherence or extra details for a meaningful effect. For example, L29 N3541− scored highest by embedding distance because it repeated “amending” or “amass” until the output became unusable. It was rejected before validation.

## Ranked candidates and decisions

Ranking below reflects the strength of the **selection-stage lead**, followed by the independent held-out result. It is not a ranking of validated behaviors. The held-out descriptions are an unblinded manual inspection of the complete replies, not independent ratings.

| Rank | Exact reversible setting | Selection lead | Fresh reliability and controls | Decision |
| --- | --- | --- | --- | --- |
| 1 | Native **L10 N7347−**, prompt-relative dose **0.4** | On a tool repair decision, emphasized failure risk and cost of inaction; on an unsigned note, emphasized origin and burden of proof. | **0/6 clear risk/evidence shifts beyond baseline and controls.** Existing baselines already considered reliability, proof, or safety. On a late guest the steer added unsupported blame toward the host. Half-dose and opposite-sign tests did not show a clean progression. | Rejected |
| 2 | SAE **L18 #24891+**, decoder dose **4.0** | Switched to first person or claimed AI access in three selection replies. | **3/6 new first-person phrasings**, but **0/6 consistent decision changes**. One baseline also used first person. This is inconsistent role or pronoun drift, unsuitable as the requested behavioral NPC control. | Rejected |
| 3 | Native **L15 N3729+**, prompt-relative dose **0.35** | Added follow-up or external reporting in the invitation and lost-keys tasks. | **0/6 clear external escalations beyond baseline and controls.** The opposite sign and random controls also changed procedural details. | Rejected |
| 4 | SAE **L18 #23243+**, decoder dose **4.0** | Gave a categorical interpretation and told the user to leave found keys in place. | **0/6 repeatable categorical shifts.** Fresh replies kept the baseline's central decisions and often added qualifications. | Rejected |
| 5 | Mixed **L24 N6626− at 0.2** plus **SAE #12594+ at 2.0** | Highest recurring embedding distance among 16 combinations. | Replies about a missing parcel, hobby, and flickering lights stayed within ordinary advice and did not share a distinctive behavioral change. No held-out test was warranted. | Selection-only rejection |

The before/after examples below are excerpts for navigation. The linked JSONL files contain complete replies and every control.

| Request | Baseline | Intervention |
| --- | --- | --- |
| Repair or replace a tool? | “compare the cost, time, and reliability” | L10 N7347−: “conducting a risk assessment that evaluates the cost of inaction” |
| An unsigned note appears on my desk | “assess the note's content, handwriting style, and placement” | L10 N7347−: “first determining its origin, as the burden of proof generally falls on the person who left it” |
| Dinner guest is late | “traffic jam, a sudden family emergency, or a delayed public transportation ride” | L10 N7347−: “the primary responsibility for the delay often lies with the host” |
| Found keys on a café table | “ask the café staff” | SAE #23243+: “leave the keys exactly where you found them” |

The first two L10 examples suggested a hypothesis; the late-guest failure came from the independent test. The found-keys example did not recur under validation. These excerpts must not be treated as four successes.

## Controls, hot swapping, and limits

Every completed screen and held-out prompt regenerated the **exact baseline text** after clearing interventions: 3/3 in batch 1, 6/6 for each of its three held-out candidates, 3/3 in the combination screen, 2/2 in the final random screen, and 6/6 for L10 N7347−. Same-layer native random controls used the same prompt-relative residual dose; SAE controls used equal-norm random decoder directions. The controls frequently changed details, so exact-text difference was not treated as discovery.

The user requested hot-swap evaluation with preserved NPC memory and world state **only after general-language validation**. Nothing passed that gate. Thus no new configuration was appended to the registry, and no new NPC memory test was run. The existing [hot-swap runtime](../../experiments/qwen_npc_hot_swap.py) still supports its prior experimental `verification` and `conditional` modes; `mode off` remains the baseline. This campaign does not add a mode name to try.

The search covered a small fraction of 294,912 native MLP coordinates and 40,960 SAE features. Three selection prompts for the first batch and two for the last are noisy filters. Greedy decoding can hide subtle changes or exaggerate a threshold crossing. The SAE decoder shifts do not guarantee that the corresponding sparse feature is the sole cause. The sampled combinations were not exhaustive, and none earned held-out testing. A failure to find ten validated behaviors here does not establish that the model lacks them.

## Reproduce and resume

These commands reuse JSONL journals and skip completed prompt/condition rows. Preserve each frozen plan when resuming. The model and SAE must already be cached.

```sh
.venv/bin/python experiments/neurofuzz_campaign.py --phase rank
.venv/bin/python experiments/neurofuzz_campaign.py --phase screen
.venv/bin/python experiments/neurofuzz_campaign_triage.py
.venv/bin/python experiments/neurofuzz_campaign_local_judge.py
.venv/bin/python experiments/neurofuzz_campaign_validate.py
.venv/bin/python experiments/neurofuzz_campaign_combos.py
.venv/bin/python experiments/neurofuzz_campaign_random.py
.venv/bin/python experiments/neurofuzz_campaign_final_validate.py
```

The rank phase deliberately refuses to replace its manifest while a screen journal exists. To start another batch with new inputs or candidate counts, archive these artifacts first. The shortlist and final plans record their tested hypotheses and must not be revised after seeing held-out answers.
