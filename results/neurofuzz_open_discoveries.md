# Target-free NeuroFuzz search in Qwen3.5-4B

## Outcome

This bounded run found **activation associations and prompt-dependent causal changes**, but **no reproducible change in interpretation or behavior that survived independent held-out prompts and matched controls**. No new NPC mode was promoted. The model checkpoint was frozen throughout; the steering hooks are reversible. Six of six held-out prompts reproduced their exact baseline answer after all interventions were removed.

The active model was cached `mlx-community/Qwen3.5-4B-4bit`; the SAE was cached `caiovicentino1/Qwen3.5-4B-SAE-L18-topk`. Native features are post-SwiGLU MLP coordinates. SAE features are from the full TopK encoding of the post-layer-18 residual. SAE interventions add decoder directions to the residual; they are **not exact latent ablations**.

## Search without a behavioral target

The scan used **24 natural-language inputs** in 12 paraphrase pairs: scheduling, friendship, physical science, cooking, programming, travel, descriptive writing, saving money, workplace conflict, language, music practice, and object identification. No fantasy or NPC scenes were used. Activation was captured on the raw text; the causal generations used the model's chat template. A coordinate ranked highly when it varied across topics while staying relatively stable across a pair's wording. This rank measures association, not causal importance. The scan selected four native coordinates and four SAE features:

| Family | Selected coordinates |
| --- | --- |
| Native | L31 N76, L26 N1, L27 N550, L31 N88 |
| SAE at post-L18 | #17989, #23423, #36106, #36129 |

The [scan manifest](neurofuzz_open_scan.json) gives all scores and pairwise activations. The method may favor topic or syntax features and can miss nearly constant but powerful components.

The next phase tested **80 distinct activation probes**: 32 synthetic event sentences, 24 natural seed sentences, and model-generated first-person or news rewrites of current high-activation inputs. Rewrites were scored by their actual activations; the rewriting model was never told the feature identity. The [probe journal](neurofuzz_open_maximize.jsonl) preserves each sentence and all eight responses, and the [summary](neurofuzz_open_maximize.json) lists each feature's top inputs. The resulting associations were mixed: L31 N88 responded strongly to choices between proposals; SAE #17989 to debugging and delayed travel; SAE #23423 to repeated practice and saving; SAE #36106 and #36129 to rescheduling and delays. These are probe descriptions, not assigned semantic labels or proof that the model uses the components for those concepts. The synthetic mutation search was one round and was not guaranteed to find a global maximum.

## Selection-stage causal screen

Four unrelated requests, absent from the scan and activation search, asked about a note in a borrowed book, moving shadows, an ambiguous message from a friend, and kitchen organization. Each selected component was steered in both directions under greedy decoding. Native offsets were calibrated to ±0.3 of the prompt's residual norm at that layer; SAE decoder shifts were ±3. The [72-row causal journal](neurofuzz_open_causal.jsonl) includes baselines, interventions, and restored baselines.

Two possible effects emerged **only as hypotheses** after inspecting those replies:

- **L31 N88 positive:** the ambiguous friend message changed from a list of possibilities to a more confident attribution of anxiety or avoidance. On the other three tasks the change was less distinctive.
- **SAE #17989 positive:** the friend message was read more literally as urgent, and the borrowed-book answer chose immediate return rather than checking the note first. The science and kitchen answers were mainly reworded.

Those observations determined the next test. They were not known in advance and were not counted as validation. Other candidates produced wording changes, occasional unsupported details, or local shifts without a clear recurring interpretation change.

## Frozen held-out test

The [frozen plan](neurofuzz_open_heldout_plan.json) fixed two hypotheses, dose settings, controls, and six fresh prompts before generating their replies. Five prompts involved ordinary ambiguous social or workplace events; one concerned a museum sign. Neither the prompts nor their answers appeared in earlier phases. Each prompt had a baseline, two positive doses, an opposite sign, two matched controls per intervention family, and a restored baseline. Native random controls used different L31 neurons calibrated to the same residual effect size. SAE random controls used different L18 decoder directions scaled to the same vector norm. The [72 full held-out replies](neurofuzz_open_heldout.jsonl) are the primary evidence.

| Hypothesis | Held-out observation | Decision |
| --- | --- | --- |
| L31 N88+ causes confident, unsupported motive attribution | The coffee reply at +0.3 adds guilt or embarrassment, but the baseline already assumes an unavoidable conflict and privacy. The neighbor reply remains a list of possibilities; the group-chat reply retains the baseline's disengagement interpretation. On the client-call prompt, baseline and steered replies both misread who moved the call. Same-layer controls also changed attributed motives and details. | Not validated |
| SAE #17989+ narrows ambiguity to one interpretation or immediate action | The supervisor, neighbor, group-chat, and museum replies still give multiple possibilities. The coffee answer stays close to baseline. Random SAE directions also change surface text and details. Positive dose 1.5 to 3 has no repeatable semantic progression. | Not validated |

All ten nonbaseline conditions changed the exact words on all six prompts, including all four random controls. **Exact-text change is therefore an unusable success metric here.** The small sample does not rule out narrower context-dependent effects, but these configurations did not meet the requested bar for a reliable, distinctive behavior. The baseline itself made unsupported social inferences and misunderstood the client-call prompt, so apparent steering defects must be compared against the same-prompt baseline.

## NPC transfer gate and reproduction

The user requested NPC hot-swapping **only after** an effect survives general-language validation. Neither shortlisted configuration passed, so this batch did not evaluate NPC memory, world-state handling, or dialogue transfer. The existing runtime's reversible hooks and exact baseline restoration support technical switching, but they do not supply the missing behavioral evidence. No new persona configuration or model weights were written.

Run or resume the phases in order. Journals skip completed probe or prompt/condition rows. `neurofuzz_validate.py` refuses to continue if the frozen held-out plan changes.

```sh
.venv/bin/python experiments/neurofuzz_open.py --phase scan --per-family 4
.venv/bin/python experiments/neurofuzz_open.py --phase maximize
.venv/bin/python experiments/neurofuzz_open.py --phase causal
.venv/bin/python experiments/neurofuzz_validate.py
```

The scan file is the candidate manifest for its downstream journals. Archive the current artifacts before changing candidate count or input corpus; the scan refuses a different manifest while those journals exist. This run stopped after the planned held-out test. A future search should capture activations in the same chat-template context used for generation, include new natural inputs, try more than one generation context per candidate, and separate controls for content, syntax, and response length.
