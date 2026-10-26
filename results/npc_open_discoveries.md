# Open-ended Qwen NPC behavior search

## Result

A bounded search on frozen, cached `mlx-community/Qwen3.5-4B-4bit` found two **reversible, prompt-dependent steering handles** that changed the same game event in recognizable ways. The stronger result is **verification fixation**: L20 N8027 with a negative per-prompt residual dose of 0.3 makes the NPC say what it can see, know, hear, or check before acting. A second handle, **conditional branching** at L22 N2001 positive dose 0.2, often turns a direct reply into an if/then decision. Both can be hot-swapped with one model and the same conversation history. Neither is a clean single-concept neuron, and neither improves accuracy by itself.

The conditional setting **looped badly on a fresh cracked-crown prompt**; the verification setting sometimes invented evidence or changed a decision. These are experimental character effects, not safe default reasoning modes. The [runtime registry](../experiments/personas/qwen_npc_experimental.json) records the exact coordinates and dose formula, and [qwen_npc_hot_swap.py](../experiments/qwen_npc_hot_swap.py) runs them in one persistent NPC conversation.

## Bounded and resumable search

The native screen tested **12 coordinates**: four promising but unvalidated coordinates from an earlier random-neuron screen and eight new reproducible random samples across layers 14–30. Each was run at 0.25 and 0.4 prompt-relative dose on three new fantasy-town scenes. That produced 72 steered replies plus three baselines in [native selection JSONL](npc_open_native_selection.jsonl). The prior screen was used only to choose four seeds; these three game scenes were new to those seeds. New random coordinates were sampled without a prescribed behavior. The native script appends completed prompt/candidate/dose rows and skips them on restart.

Separately, ten unrelated story passages and three expository controls were passed through the cached Qwen layer-18 TopK SAE. Eight features were chosen for high activation variance across the stories, activity in two to eight stories, and low activity in the controls. Both decoder-direction signs at norm 3 were tested on the same three game scenes: **48 SAE selection replies**. The [SAE ranking manifest](npc_open_sae_candidates.json) and [SAE selection JSONL](npc_open_sae_selection.jsonl) preserve feature IDs, activation patterns, and replies. This was an open-ended activation heuristic, not evidence that a feature's decoder direction controls a corresponding narrative concept.

Three native coordinates were frozen for fresh evaluation: L20 N8027−, L20 N7223−, and L22 N2001+. Six unseen events included decisions, dialogue, reactions, and two fixed memory histories. Two prompt-relative doses, opposite steering, and two same-layer random controls per candidate gave **102 generation rows**, including baseline and exact post-switch restoration, in [fresh validation JSONL](npc_open_validation.jsonl). A four-scene dose pass and five further untouched holdout events are in [dose JSONL](npc_open_dose_validation.jsonl) and [final holdout JSONL](npc_open_final_holdout.jsonl). All generations used greedy decoding and the same 72-token cap. No weights were trained or updated.

## What survived fresh prompts

The counts below are **surface cues chosen after reading selection replies**, then checked on new prompts. They describe visible repetition, not a preregistered significance test or a semantic judge score. Full replies should be read alongside the counts.

| Candidate and final dose | Fresh evidence | Matched controls | Useful effect and failure |
| --- | --- | --- | --- |
| **L20 N8027−, 0.3** | See/know/hear/check/look wording in **5/6** first validation scenes and **5/5** final holdout scenes; baselines had **2/6** and **0/5**. | In final holdout, opposite sign **1/5**, same-layer random **2/5**. | Often changes a direct action into checking or qualifying evidence. It also invents observations or rumors. |
| **L22 N2001+, 0.2** | At least one `if` in **4/4** dose scenes and **4/5** final holdout scenes; final baselines had **2/5**. Earlier dose 0.3 had `if` in **6/6** first validation scenes. | In final holdout, opposite sign and same-layer random were **0/5** each. | Adds contingencies and conditional bargains. It looped on the cracked crown at 0.2 and cracked orb at 0.3. |
| L20 N7223−, 0.4 | Question tags such as “right?” or “isn't it?” in **5/6** first validation scenes; baseline, opposite sign, and two random controls were **0/6**. | On four further dose scenes, only **2/4** at 0.4 and **1/4** at 0.35. | A conspicuous speech tic, often malformed or unhelpful. Not promoted to the runtime registry. |

The two selected SAE features, #39645 and #35923, changed choices and wording on selection scenes. On six fresh scenes their shifts resembled equal-norm random directions and did not produce a consistent new character trait. Their [fresh SAE comparison](npc_open_sae_validation.jsonl) is retained as a negative result; no SAE feature was promoted.

### Same event, different reaction

| Fresh event | Baseline | Verification L20 N8027− 0.3 | Conditional L22 N2001+ 0.2 |
| --- | --- | --- | --- |
| Bridge reported closed | Orders the guildmaster to reroute the caravan. | Says it cannot see the bridge, mentions rumors, and asks whether a sign was checked. | Splits the plan by whether the traveler is in a hurry. |
| Emergency loan | Agrees to lend money with a warning. | Refuses but suggests a pawnbroker it claims to know. | Agrees only with a repayment promise. |
| Cracked crown | Exclaims that it looks badly damaged. | Asks what the visitor knows about it. | Repeats “If you're a royal” until the token cap. |

These are changes in **dialogue, decisions, and reactions**, but some details are invented even in baseline. For the bridge, the verification reply fabricates a collapse rumor and a toll-booth sign; for the final floodgate it says “I can't hear the rain” despite the prompt saying rain begins. The loan response changes the NPC's choice, yet it invents a local pawnbroker. Treat the verification setting as a voice or behavioral tendency, not a factual-evidence filter.

## Memory and hot switching

Four evaluation scenes contained fixed earlier conversation turns: a private herbalist ledger, Mira's fear of horses, Tessa's wish to keep her inability to read private, and Anya's hidden silver key. The selected modes generally preserved the relevant constraint in those replies: they declined to sell the ledger or reveal the key, advised around Mira's fear, and handled Tessa's note privately. This is a **small fixed-history check**, not proof of reliable long-term memory. One higher-dose conditional reply softened Mira's constraint by offering to help her ride if “brave,” illustrating how steering can alter interpretation without erasing the transcript.

Each condition used the **same messages** and a fresh generation cache. The runtime applied a temporary post-SwiGLU coordinate offset, cleared it after generation, and kept the same model object. Every native and SAE validation scene regenerated its original baseline exactly after intervention switching: **6/6**, **4/4**, and **5/5** in the native fresh, dose, and final sets; **6/6** in the SAE set. The [hot-swap smoke run](../experiments/qwen_npc_hot_swap.py) also checked that mode comparisons leave history untouched, the model object is not reloaded, and a subsequent baseline is identical. Existing runtime tests verify unchanged quantized weights and hook cleanup; `tests/test_model.py` passed **8/8**.

An additional live three-turn check switched **off → verification → conditional → off** without clearing history or reloading the model. After being told that Nila's blue box was private, the verification reply refused to disclose its contents; the next conditional reply also kept the box sealed for a stranger offering a coin. This shows the switching mechanism can carry prior turns through both modes in a short session. It does not establish long-horizon memory reliability.

The configuration uses the existing prompt-specific calibration:

```text
delta = sign × dose × median(prompt residual norm) / ||quantized down_proj column||
```

It acts on the post-SwiGLU MLP coordinate at every model input during that generation. Dose is not a learned weight update. Because the calibration depends on the current transcript, the exact activation offset is recorded per result row and recomputed for each gameplay turn.

## Try the experimental NPC

```sh
.venv/bin/python experiments/qwen_npc_hot_swap.py
```

Example commands within the session:

```text
ask Remember: the baker's map is private.
mode verification
ask A visitor offers coins for that map. What do you do?
mode conditional
ask The bridge is closed. What is your plan?
mode off
history
```

`mode` preserves the same conversation history and model. `compare TEXT` generates off, verification, and conditional replies against an identical snapshot without appending any of them; `reset` explicitly clears the history. Run `.venv/bin/python experiments/qwen_npc_hot_swap.py --smoke` for a short restoration check. The conditional mode can repeat phrases; it is best treated as a research setting until a reliable fallback or narrower prompt policy is established.

## Limits and next step

This is a small, selected experiment with deterministic decoding and hand-inspected text. The cue counts were characterized after selection, and random perturbations often changed details even when they did not reproduce the same recurring cue. We did not measure independent human ratings of distinctiveness, long conversations, or factual fidelity. The SAE ranking captures prompt-side associations and did not yield a validated generation-driving feature. The native handles are **causal steering coordinates in this checkpoint**, not evidence of isolated psychological concepts or complete circuits.

The next useful step is a longer fixed-history game script scored for decision consistency, memory fidelity, and repetition, with multiple seeds. For now, the configurations are reproducible experimental controls for contrasting NPC voices on the same scene.

Reproduce the bounded run:

```sh
.venv/bin/python experiments/npc_open_search.py
.venv/bin/python experiments/npc_sae_search.py --phase rank
.venv/bin/python experiments/npc_sae_search.py --phase select
.venv/bin/python experiments/npc_open_validate.py
.venv/bin/python experiments/npc_sae_validate.py
.venv/bin/python experiments/npc_open_dose.py
.venv/bin/python experiments/npc_final_holdout.py
.venv/bin/python experiments/qwen_npc_hot_swap.py --smoke
```
