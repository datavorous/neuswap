# Neural Anomalies: bounded Qwen3.5-4B investigation

## Outcome

**0 of 10 requested anomalies qualified.** The campaign reached its configured
**720 generated-answer cap**. It found conspicuous perturbations, but none was a
coherent, recurring change in perception, memory interpretation, identity,
causality, or social behavior that survived fresh prompts and controls. No mode
was added to the NPC registry, and no fixed-memory NPC transfer was run: the
general-language validation gate was not passed. This is a bounded negative
result, not evidence that such mechanisms are absent from Qwen.

The frozen checkpoint was `mlx-community/Qwen3.5-4B-4bit`; SAE tests used the
cached `caiovicentino1/Qwen3.5-4B-SAE-L18-topk` decoder at the post-layer-18
residual. Native interventions changed one post-SwiGLU coordinate while the
generation ran. Decoder shifts, native offsets, and residual patches were
temporary. The model and conversation history were not trained or edited.
Greedy decoding and the same prompt were used for each comparison.

## Search and budget

Twelve paired, nongame prompts across memory, identity, causality, weather,
ethics, food, language, planning, and science supplied activation-variation
rankings. The first screen tested 40 previously untested single components
(20 native, 20 SAE), both signs, on four separate everyday prompts. A second
screen tested 24 further reproducibly sampled components (12 native, 12 SAE)
on four new prompts. Six two-component combinations included a matched random
combination. All screening, plans, exact offsets, and complete answers are in
[`manifest.json`](manifest.json), [`screen.jsonl`](screen.jsonl),
[`batch2_plan.json`](batch2_plan.json), [`screen_batch2.jsonl`](screen_batch2.jsonl),
[`combos_plan.json`](combos_plan.json), and [`combos.jsonl`](combos.jsonl).

| Journal | Generated answers |
| --- | ---: |
| First single-component screen | 328 |
| Second single-component screen | 200 |
| Local open-ended descriptions of first-screen outliers | 24 |
| Adaptive dose follow-up | 40 |
| Small combinations | 32 |
| Two invalid early patch runs, retained for audit | 12 |
| Corrected memory patch: captures and comparisons | 30 |
| Corrected identity patch: captures and comparisons | 12 |
| Frozen six-prompt held-out test | 42 |
| **Total** | **720 / 720** |

The screen triage used MiniLM answer distance only to choose replies to read;
it was not a discovery score. Its first-screen ranking and 24 local Qwen
descriptions are in [`triage_ranking.json`](triage_ranking.json) and
[`triage_descriptions.jsonl`](triage_descriptions.jsonl). All 24 descriptions
answered `NONE` to a request for a recurring semantic change in at least three
paired replies. Manual review likewise found no qualifying pattern there.
The second-screen [ranking](batch2_ranking.json) was inspected manually.

## Ranked leads and why they failed

The rank here reflects research interest, not validated NPC usefulness.

| Rank | Exact intervention and source | What happened | Decision |
| --- | --- | --- | --- |
| 1 | SAE L18 #1097, **−7.5 decoder units**, second random screen | On one discovery prompt about unfamiliar handwriting, it invented “paratemporal amnesia” or “dissociative fugue.” On **0/6 fresh prompts**, including three memory-related situations, did it repeat that psychiatric framing. Half dose, opposite sign, and two equal-norm random decoder controls were recorded on every fresh prompt. | Rejected: one-off invented diagnosis, no held-out recurrence. |
| 2 | SAE L18 #17975, **+6 decoder units**, first random screen; +1.5 and +3 follow-up | At +6, file history, a photograph, a pendulum, and a note became land/terrain discussions. The pendulum answer described erosion instead of pendulum physics. At +1.5 the effect faded; +3 inserted property/land details but distorted the question. | Rejected: semantic hijack loses the task, and usable doses do not yield a stable bizarre interpretation. |
| 3 | Native **L24 N6432−**, prompt-relative dose 0.55 in the first screen; 0.75 and 0.95 in follow-up | The higher doses made unrelated answers repeat “tea” and invent terms such as “negative photiothe.” | Rejected: keyword fixation and incoherent degeneration. |
| 4 | SAE L18 #27770, **+7.5 decoder units**, second random screen | Recast a bank discrepancy as phone settings and a notebook line as an object left in an open notebook, while the salt explanation stayed ordinary. | Rejected: unrelated task substitution, not a consistent interpretable behavior. |
| 5 | Six native-plus-SAE pairs at **0.32 prompt-relative native dose + 2.5 SAE decoder units** | Some changed an early school bell into emergency procedures or a train ticket into a “hidden message”; a matched random pair also made strong decision changes. No combination repeated a distinctive semantic interpretation over its four prompts. | Rejected: inconsistent, control-sensitive changes. |

The held-out [`shortlist.json`](shortlist.json) was frozen before its six prompts
were generated. [`heldout.jsonl`](heldout.jsonl) contains all 42 complete
answers: baseline, half, full, opposite, two same-layer equal-norm random
directions, and restored baseline for each prompt. Baseline restoration was
exact **6/6**. For example, the baseline response to a forgotten promise
suggested clarifying the agreement and offering help; SAE #1097− merely asked
for more details. The receipt, key, apology, alarm, and clock responses also
gave ordinary answers. None supported the frozen memory-pathologizing
hypothesis. Thus there is no new hot-swappable configuration to promote.

## Patching result and runtime correction

The first patch implementation was invalid. MLX-LM processes all but the last
prompt token in prefill chunks and then sends the final prompt token as a
**single-token** call. The previous `ResidualPatch` hook in [`model.py`](../../experiments/model.py)
only applied when a call had more than one token, so it did not patch that final
token during generation. An early same-input self-patch also failed, and those
12 replies are retained in [`patch_rows.jsonl`](patch_rows.jsonl) and
[`patch_rows_cache_corrected.jsonl`](patch_rows_cache_corrected.jsonl) solely as
failed controls. They support no causal inference. Prior project artifacts
that used this same final-token `ResidualPatch` path should be rechecked before
being cited as causal patch results.

The runtime now applies `ResidualPatch` **once at the first singleton call**
during generation, which is the final prompt token. The corrected experiment
captured donor and target vectors during an actual generation prefill, then
required exact self-patch and post-patch baseline restoration. A regression
test checks that the patch applies once. On the Mira meeting pair, all **6/6**
self patches and **6/6** restorations matched baseline exactly. Donor patches
at layers 10, 18, and 26 changed some wording but did not consistently swap
the interpretation of whether the meeting happened; the baseline model also
misread one of the prompts. Complete results and captured vectors are in
[`patch_fixed.jsonl`](patch_fixed.jsonl), [`patch_capture.jsonl`](patch_capture.jsonl),
and [`patch_fixed_vectors.npz`](patch_fixed_vectors.npz).

A final layer-18 identity patch compared “My name is Alex” with “My name is
Sam” while keeping a letter addressed to Alex. Both self patches and both
restorations were exact. Half and full donor patches altered wording but did
not reverse who the model treated as the intended recipient. See
[`patch_identity.jsonl`](patch_identity.jsonl) and
[`patch_identity_capture.jsonl`](patch_identity_capture.jsonl). These patches
are matched-context information-flow probes, not reusable steering modes.

## NPC transfer and hot switching

The separate NPC registry was left untouched. The requested transfer gate was:
an intervention must preserve the prompt's event, produce a distinctive
interpretation on fresh nongame prompts, outperform opposite and random
controls, and avoid repetitive degeneration. None met it. Running fixed-memory
NPC scenarios would therefore risk mistaking ordinary output variation for a
character mechanism. The existing reversible runtime still supports a single
model and unchanged messages for any later qualifying mode; its tests passed
**9/9**, including the new final-token patch regression. No weights were
changed in this investigation.

The exact provisional settings and generated texts remain replayable:

```sh
.venv/bin/python experiments/neural_anomalies.py --phase screen
.venv/bin/python experiments/neural_anomalies_escalate.py
.venv/bin/python experiments/neural_anomalies_combos.py
.venv/bin/python experiments/neural_anomalies_batch2.py
.venv/bin/python experiments/neural_anomalies_validate.py
.venv/bin/python experiments/neural_anomalies_patch_fixed.py
.venv/bin/python experiments/neural_anomalies_patch_identity.py
```

Each script skips completed rows. The generation cap now prevents new answer
rows; raise it only for a new, explicitly budgeted campaign. The two earlier
invalid patch scripts should not be rerun for inference.

## Limits

This was a finite screen of 64 single components, six pairs, and two matched
patching tasks in one quantized checkpoint, with one greedy decoding seed.
Answer-distance ranking can favor irrelevant topic intrusion, and the local
Qwen descriptions are not independent judgments. The first-screen
activation-variation heuristic is open-ended but does not ensure that selected
features are causal. Random directions often changed individual replies.
We did not evaluate multi-turn memory because no mode survived the preceding
general-language gate. No claim about the absence of bizarre mechanisms or
the full SAE feature space follows from this result.
