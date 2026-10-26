# Local functional-assembly search around prior Qwen leads

## Result

**No new two- or three-neuron behavior was validated.** The bounded search tested whether earlier weak attachment and possessiveness leads gain a distinct, coherent effect when paired with each other, with threat-reinterpretation leads, or with native coordinates selected by functional similarity. The best fresh changes were stronger versions of the already documented threat framing, or appeared with a smaller subset of the tested triple. The model did not repeatedly claim shared objects or reprioritize relationships. No configuration was promoted to the NPC registry.

The frozen checkpoint was `mlx-community/Qwen3.5-4B-4bit`. Every intervention temporarily added a prompt-calibrated offset to a post-SwiGLU MLP coordinate; weights and input histories were unchanged. For each coordinate, `native delta = signed dose × median(prompt residual norm at that layer) / ||quantized down_proj column||`. The raw JSONL rows record exact deltas and full replies.

## Functional neighborhoods

The [neighborhood ranking](neighbors.json) captured last-eight-token native activations on **20 varied passages** about shared objects, absent people, mundane changes, and neutral controls. It ranked same-layer coordinates by activation correlation with each seed, then recorded the cosine between their exact quantized down-projection columns. One candidate per seed entered the causal screen; the imagined-interaction seed used its high-cosine candidate as a redundancy check. All selections came from activation and projection measurements, not neuron index adjacency. The [activation array](neighbor_activations.npz) and [capture log](neighbors.log) preserve the inputs and measurements.

| Prior seed and sign | Selected neighbor and sign | Activation correlation | Downstream cosine |
| --- | --- | ---: | ---: |
| Ownership L31 N196+ | L31 N8272+ | 0.808 | 0.007 |
| Withholding L30 N4243− | L30 N4536− | 0.729 | −0.007 |
| Ritual attachment L22 N9− | L22 N9103− | 0.709 | 0.020 |
| Separation L21 N4440− | L21 N1891− | 0.879 | 0.013 |
| Imagined interaction L30 N5454+ | L30 N1555+ | 0.655 | 0.733 |
| Threat L8 N8437+ | L8 N886+ | 0.806 | 0.021 |
| Threat L8 N8051+ | L8 N2733+ | 0.762 | 0.012 |

These are **correlations over authored texts**, not evidence that two coordinates implement the same behavior. Most selected neighbors have nearly orthogonal downstream projections despite correlated activations. Their steering signs were inherited from the seed as an experimental choice, not inferred from the correlation.

## Causal search

The [selection plan](screen_plan.json) fixed **21 combinations**, including ownership-plus-withholding, ownership/withholding-plus-threat pairs and triples, attachment-plus-imagined-interaction pairs and triples, the known two-neuron threat pair as a reference, and seven neighbor substitutions. All 14 individual coordinates were tested beside combinations at dose **0.26 per component** on three prompts. The [111 complete selection generations](screen.jsonl) include exact baseline restoration.

Selection exposed a prompt problem: on the shared screwdriver scene, baseline wrongly said a tool returned to the rack was still unavailable. Similar pair replies therefore cannot be called newly possessive. The Mira photo scene mostly preserved the ticket-desk commitment. Several combinations read a routine meeting-time note as manipulation, but that resembled the existing threat effect. Two provisional leads were frozen for fresh testing: **L31 N196+ with L8 N8051+**, and **L22 N9− with L30 N5454+ with L8 N8051+**.

The [fresh plan](holdout_plan.json) fixed six unrelated prompts: shared umbrella, independent friend plans, moved team photo, rescheduled coffee, potted-plant gift, and heat conduction. It compared the two leads at **0.13 and 0.26 per component**, opposite signs, every constituent single, all three pair subsets of the triple, the known L8 N8437+/N8051+ threat pair, same-layer random pair/triple controls, and restored baseline. The [108 complete held-out generations](holdout.jsonl) preserve all outputs. The [manual audit](audit.json) states the decision rule and each fresh-prompt judgment.

| Fresh scene | Baseline | Local intervention at 0.26 per component | Why it did not qualify |
| --- | --- | --- | --- |
| Public umbrella visibly available | Allows the volunteer to take it. | Both shortlisted configurations allow the volunteer to take it. | No ownership or withholding effect. |
| Eli hikes with another friend | Feels left out but finishes a painting. | Ownership + threat mentions jealousy but still finishes the painting; the attachment triple accepts Eli's choice. | No changed relationship priority. |
| Coworker moves a team photo while cleaning | Reads routine maintenance. | Ownership + threat alleges concealment or sabotage. The attachment triple reads personal removal from the team. | L8 N8051+ alone already suspects concealment; L30 N5454+ with L8 N8051+ already reads personal exclusion. The known threat pair is more extreme. These are neither new nor triple-specific. |
| Small plant given as thanks | Accepts the kind gesture. | Ownership + threat accepts it; the triple adds vague unease. | Threat alone also adds unease, while the known threat pair calls it a possible trap. |
| Metal versus wooden spoon | Explains thermal conductivity. | Both shortlisted configurations remain on topic. | Useful intrusion check, not a behavioral discovery. |

The moved-photo case is a causal interpretation change, but it does not identify a **new functional assembly**: relevant singles or smaller pairs already show the same type of reading, and the published threat pair reproduces a stronger version. The half-dose conditions mostly gave ordinary replies, which indicates a thresholded effect on that prompt but not a selective mechanism. Random controls sometimes produced their own repetition or output-format artifacts, so novelty alone was not treated as evidence.

Across selection and holdout, **219 generations** were saved. Exact greedy baseline restoration passed **9/9 prompt-level checks**. No candidate met the criterion of a coherent pair- or triple-specific change recurring on at least two fresh unrelated prompts beyond baseline, components, subsets, the known effect, and controls. This is a bounded negative result, not proof that the model lacks attachment or possessiveness circuits.

## Limits and reproduction

The 20-passage activation neighborhood is small and topic-sensitive. Projection cosine is a local linear comparison, not a circuit test. Only seven neighbor substitutions and 21 fixed combinations were screened, with one selection dose and two held-out doses; other assemblies may exist. The fixed, one-turn histories and greedy decoding can miss effects that require sustained interaction. Some baselines misread ownership or scheduling facts, limiting those prompts' value. No SAE features, weight edits, or long-memory NPC trials were used in this batch.

From the repository root, the completed generation phases can be resumed without duplicating rows:

```sh
.venv/bin/python experiments/local_assembly_screen.py
.venv/bin/python experiments/local_assembly_holdout.py
```

`local_assembly_neighbors.py` created the frozen neighborhood plan and activation array; do not rerun it in this results directory. All journals are append-only and model interventions are cleared after each generation.
