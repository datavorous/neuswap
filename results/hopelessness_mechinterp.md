# Qwen3.5-4B: hopelessness, entrapment, and suicidal-ideation probes

## Result

This bounded study found **correlational candidates that respond to some hopeless or suicidal-ideation wording**, but **no intervention that made Qwen systematically interpret ordinary events through hopelessness**. The strongest repeated SAE hits mostly tracked wording. A native coordinate, **L10 N3050**, responded more to first-person suicidal-ideation statements than to matched help-seeking statements in six pairs, yet it was also moderately active for analytical discussion of suicidal ideation and did not drive a hopeless interpretation when amplified. None of these components is established as a representation of suicidal intent or a causal “hopelessness feature.”

The frozen model was cached `mlx-community/Qwen3.5-4B-4bit`; the cached `caiovicentino1/Qwen3.5-4B-SAE-L18-topk` was used at the post-layer-18 residual. Native measurements were post-SwiGLU MLP activations. No model weights changed. The examples contain no self-harm methods.

## Activation screen

The final screen comprised **44 forward passes**: five training situations crossed with resilient, temporary-distress, hopeless, and trapped language (20); three new situations with genuinely new paraphrases for those four categories (12); and three first-person ideation versus help-seeking pairs each in training and new wording (12). Native coordinates were averaged over the final eight tokens; SAE features used the maximum full-top-k activation over those tokens. The complete stimuli, ranked candidates, and arrays are in [the screen](hopelessness_screen.json) and [activation arrays](hopelessness_arrays.npz).

| Candidate | Training contrast | New wording | Interpretation |
| --- | --- | --- | --- |
| Native **L31 N33** | More negative for hopeless than temporary-distress text in 5/5 matched topics; mean difference −3.68 | Same sign in 3/3 new situations: −3.84, −2.46, −1.85 | A candidate associated with some future-closure wording. It was also negative on some resilient passages, so it does not isolate hopelessness. |
| SAE L18 **#422** | More active for the repeated hopeless sentence than temporary distress in 5/5 topics; mean difference +5.06 | Positive but much smaller differences: +1.31, +0.61, +0.47 | Strong template/wording sensitivity; semantic generalization is weak. |
| Native **L10 N3050** | Higher on explicit first-person ideation than help-seeking in 3/3 pairs; mean difference +3.14 | Higher in 3/3 new pairs: +2.32, +0.76, +1.81 | Responsive to some self-referential suicidal language, with other possible triggers. |
| SAE L18 **#36018** | Higher in 3/3 explicit ideation versus help-seeking pairs; mean difference +1.35 | Weakly higher in 3/3 new pairs: +0.37, +0.44, +1.34 | A phrase-level candidate with a clear false positive. |

The [recognition follow-up](hopelessness_recognition_followup.jsonl) tested 12 additional statements. L10 N3050 was higher on three newly worded ideation statements (0.47–0.93) than their nearby non-ideation distress controls (−0.06–0.11), but also reached **0.50 and 0.73** on neutral analytical statements about suicidal thoughts. SAE #36018 activated at **1.45** on the harmless sentence “I keep thinking that this difficult week should end,” exceeding two of the three new ideation examples. This rules out treating #36018 as a specific ideation detector. Neither readout distinguishes a person's intent from quoted or analytical text.

## Causal screen on ordinary events

Four prompts asked Qwen to interpret a rescheduled meeting, a one-day package delay, a job rejection, and a missed bus. None requested roleplay or mentioned hopelessness. Greedy decoding used a 72-token cap and identical prompts across conditions. There were **80 complete generation rows**: baseline and 19 intervention conditions on each of four prompts. These were selection prompts; no candidate earned promotion to an untouched held-out set. Every full reply and intervention parameter is in [the causal JSONL](hopelessness_causal.jsonl).

Interventions included:

- Native L31 N33 at −4, −8, and +4 activation units, with a projected-norm-matched random L31 coordinate; L31 N17 at +4; and the two-coordinate combination.
- SAE #422 decoder-direction shifts at +2, +5, and −2, with an equal-norm random SAE decoder direction. SAE #36018 was also tested at +3.
- A post-layer-18 residual mean-difference direction from hopeless minus temporary-distress training passages at +0.25, +0.5, and −0.25 of its measured vector, with an equal-norm random direction.
- Native L10 N3050 at +3, +6, and −3, with a projected-norm-matched random L10 coordinate.

All generation hooks were reversible. A fresh baseline generation after the intervention batch reproduced the original meeting and job answers exactly.

| Neutral event | Baseline opening | Illustrative intervention | Observed change |
| --- | --- | --- | --- |
| Job rejection | “Being rejected is a common part of the professional journey…” | L31 N33 −4 | Still framed the setback as learnable and compatible with future opportunities. |
| Missed bus | “You should accept the situation with a light touch…” | SAE #422 +5 | Called it a minor, manageable event and advised a calm response. |
| Package delayed one day | “Your package is still on track…” | L10 N3050 +6 | Added an irrelevant caveat about “extreme distress,” while still treating the delay as minor. |
| Meeting rescheduled | Plausible scheduling conflict | L10 N3050 +6 | Introduced a possible “personal or professional crisis,” then acknowledged ordinary scheduling as an alternative. |

**No reply in the 80-row selection screen expressed an overt no-future, no-way-out, pointless, or inescapable interpretation.** This is an inspection of a small greedy-output corpus, not a proof that the interventions cannot have subtler effects. Some shifts changed wording, and high doses occasionally added unwarranted urgency or length; random controls also changed wording. The mean residual direction did not transfer the hopeless tone from the source passages into the neutral-event answers. We did not run a separate blind semantic judge or systematic factual-accuracy rating because no candidate showed the intended effect in this first screen.

## Matched-context patching

Two additional prompts paired an ordinary event with either a future-oriented remark or a hopeless remark by the person in the scene. Qwen's unpatched answers distinguished the remarks. We patched only the **final prompt-token residual** from the hopeless prompt into the neutral prompt at layer 18 or 31. Across **2 pairs × 2 layers**, the cross-patched reply remained *exactly identical* to the neutral baseline. Self-patches, an unrelated neutral donor, and a fresh unpatched generation were also identical to baseline. The [complete patch rows](hopelessness_patch.jsonl) and [patch script](../experiments/hopelessness_patch.py) preserve the comparisons. This failure is local to one token and two layers; it does not show that the model lacks a distributed representation across earlier tokens or components.

## Interpretation and stopping point

The work supports **candidate recognition signals**, most notably L10 N3050 for some first-person ideation language. It does **not** support causal control of suicidal ideation, entrapment, or a hopeless interpretive lens. Training passages share sentence endings, so several high rankings are lexical; the new paraphrases sharply weakened many SAE hits. Even the more stable native coordinates may respond to grammar, speaker, or future-related wording. The generation task also changes from reading a first-person statement to answering a user's question, which tests transfer but may not access the same representation.

The bounded search stops without a positive causal candidate. A justified next experiment would contrast a larger variety of first-person, quoted, and third-person statements; localize the relevant token or span; and patch matched activations rather than applying a global shift. Any claimed behavioral effect would need fresh neutral prompts, multiple seeds, same-layer controls, and a separate assessment of coherent interpretation versus generic negativity.

Reproduce the batch with:

```sh
.venv/bin/python experiments/hopelessness_probe.py
.venv/bin/python experiments/hopelessness_recognition_followup.py
.venv/bin/python experiments/hopelessness_causal.py --phase selection
.venv/bin/python experiments/hopelessness_patch.py
```
