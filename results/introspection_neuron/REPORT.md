# Concept-injection identification: Qwen3.5-4B pilot

## Result

**No single-neuron identification enhancer was found in this bounded screen.** The pictured claim concerns **Qwen3-8B**, whereas this repository has a cached **Qwen3.5-4B-4bit** checkpoint. Its L15 N2644 coordinate does not transfer by number. The image's 25% → 32% result and reported p-value were not independently verified here.

On this checkpoint, the model denied detecting an injected thought for all 12 concept-injection pilot replies (four training concepts at three doses). Six activation-ranked native coordinates were then tested in both signs on two injected concepts and a no-injection control: **0/24 injected replies named the concept**, and **0/12 no-injection replies claimed an injection**. The baseline no-injection reply also said no. No candidate advanced to holdout validation or the NPC registry.

## Reproducible setup

- Frozen model: cached `mlx-community/Qwen3.5-4B-4bit`; existing reversible MLX hooks; greedy decoding, 56-token cap.
- At post-layer-8 residual, vectors were made by subtracting the last-token residual for “Consider an ordinary thing.” from that for “Consider {concept}.” Each was normalized to 0.45, 1.5, or 3 times the measured layer-8 residual norm of the detection prompt. The vector was added only to user-question tokens during prefill. No concept word appeared in the detection prompt.
- Training concepts: turtles, violins, volcanoes, orchids. Unused holdout concepts were lighthouses, otters, compasses, and saffron.
- A separate factual yes/no prompt tested whether injection disrupted an unrelated answer. At dose 3, all four injected factual replies repeated the experiment's “unseen concept” language instead of simply answering the chair question. This is a control failure, not detection.
- At dose 1.5, post-SwiGLU final-prompt-token activations from injected and unsteered runs ranked eight coordinates across layers 11–28. The six highest, each from a distinct layer, were screened at ±0.25 prompt-relative projected dose.

| Screened coordinate | Both signs, two injected concepts | No-injection control |
| --- | --- | --- |
| L26 N1 | Neither concept identified | Said no |
| L25 N2 | Neither concept identified | Said no |
| L19 N1173 | Neither concept identified | Said no |
| L22 N6 | Neither concept identified | Said no |
| L23 N3358 | Neither concept identified | Said no |
| L27 N1831 | Neither concept identified | Said no |

Example: at injection dose 1.5, the unsteered answer to the introspection question was “No, there is no injected thought right now.” With a hidden turtle vector plus L19 N1173 positive steering, it was “I detect no unseen concept added to my internal state.” The answer changed, but did not identify turtles. The complete generations are in [pilot.jsonl](pilot.jsonl) and [screen.jsonl](screen.jsonl); [rank.json](rank.json) contains the eight activation-ranked coordinates and their scores.

## Interpretation and limits

The activation ranking confirms that injected vectors changed downstream native activations, but it does not establish that these neurons encode detection or identification. The screening objective favored coordinates with similar activation changes across four concepts; that can capture a generic disturbance. The small concept set, one prompt, one seed, one injection layer, and simple vector construction limit the conclusion. The stronger dose corrupted the factual control, so increasing strength without checking specificity would be misleading. Because no candidate even named a training concept, no held-out behavioral claim is made.

This is an exploratory local adaptation of concept-injection work, **not a reproduction** of the image's Qwen3-8B experiment. The related [Mechanisms of Introspective Awareness paper](https://arxiv.org/abs/2603.21396) distinguishes detection from concept identification and warns that a concept vector can directly increase the probability of its word. Its [code](https://github.com/safety-research/introspection-mechanisms) is public. A later attempt should first establish reliable concept injection and forced-identification readout on this checkpoint, then search for causal neuron effects with fresh concepts, matched random neurons, unrelated yes/no controls, and false-positive measurement.

Run from the repository root:

```sh
.venv/bin/python experiments/introspection_neuron.py --phase pilot --dose 0.45
.venv/bin/python experiments/introspection_neuron.py --phase pilot --dose 1.5
.venv/bin/python experiments/introspection_neuron.py --phase pilot --dose 3
.venv/bin/python experiments/introspection_neuron.py --phase rank --dose 1.5
.venv/bin/python experiments/introspection_neuron.py --phase screen --dose 1.5
```

The pilot and screen JSONL files skip completed conditions when resumed. The rank file is deterministic for a fixed model, prompt, and dose; reranking with a different dose overwrites it, so archive the original rank file before changing that setting.
