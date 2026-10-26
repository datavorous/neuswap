# Target-free follow-up: no new validated behavior

## Result

A bounded screen of **16 previously untested layer-18 SAE decoder directions** and **12 previously untested native MLP neurons** on frozen `mlx-community/Qwen3.5-4B-4bit` did **not** find a new recurring, coherent behavior. The raw outputs contain some unusual single replies, but the same intervention did not reproduce a recognizable change across the unrelated screen prompts. Nothing from this batch was added to the experimental NPC registry.

## Search

The [SAE plan](plan.json) selected the next unused features by activation variability across 20 previously saved varied passages, with moderate prevalence and dissimilar activation profiles. It excluded the earlier auditory, book, and plant leads. Each decoder direction was added or subtracted at dose 3 from the post-layer-18 residual on four new prompts: a disputed receipt, an unexplained late arrival, an empty sealed envelope, and an echo explanation. [All 136 replies](screen.jsonl) include a same-prompt baseline and exact post-intervention restoration.

The [native plan](native_plan.json) selected 12 new coordinates from the saved 20-passage MLP activation array, at most two per layer. Each coordinate was tested with positive and negative prompt-calibrated dose 0.35 on four other prompts: late package delivery, a found key, a small apology, and steam above hot tea. [All 104 replies](native_screen.jsonl) include baseline and restoration. For each coordinate, the numeric offset is saved per reply and was computed as `sign × 0.35 × median(prompt residual norm) / ||down_proj column||`. All weights remained frozen; generation used greedy decoding and a 72-token cap.

## Interesting screen observations, and why they were not promoted

| Intervention | Screen observation | Why it is insufficient |
| --- | --- | --- |
| SAE #40141+ | Turned the late friend's unexplained arrival into “weighing the pros and cons” and the empty envelope into a choice about trusting one's gut. | The receipt and echo replies stayed ordinary; the first two also became vague or less grounded. No recurring decision policy emerged. |
| Native L31 N162− | Treated a plain found key as a potential high-security access risk. | The late delivery, apology, and steam replies remained close to baseline. Other unrelated native offsets also made the key a security concern, so this is not a selective effect. |
| Native L30 N358+ | Used “pass” repeatedly in the key, apology, and steam replies. | This is a wording tic, not a changed interpretation or useful behavior. |
| SAE #38254+ | Suggested lifting a “lid” on an envelope, changing its object identity. | A single nonsensical substitution; no similar pattern across the other prompts. |

The baseline itself treats the empty sealed envelope as a mystery trope. That prompt is especially prone to invented significance, so isolated mystery language under a shift is weak evidence. Most directions made small changes in phrasing, logistics, or speculative details. Some caused factual or object errors. We did not choose a candidate for independent held-out testing because none passed the screen's cross-prompt criterion. Thus **zero new behaviors are validated** here. This does not show that the model lacks further useful handles; it only bounds this particular search strategy and sample.

Both phases are resumable without duplicating completed prompt/condition rows:

```sh
.venv/bin/python experiments/open_cool_search.py screen
.venv/bin/python experiments/open_cool_native.py screen
```

Their plans are frozen. All **8/8 prompt-level baselines** regenerated exactly after the interventions were removed. The batch saved **240 complete generations**.
