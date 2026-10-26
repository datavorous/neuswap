# Emotional attachment search in frozen Qwen3.5-4B

## Result

**No neuron or neuron pair was validated as a control for unusual emotional attachment.** The search found isolated striking lines, but they did not recur when the same coordinates were tested with practical tasks or established relationships. No intervention was added to the NPC registry.

Here, the intended effect was more than affectionate wording: the character would repeatedly redirect attention or decisions toward one named person even when that person was safe, had made a choice, and a separate commitment needed attention. A candidate would need to exceed the same-prompt baseline and its constituent single-neuron interventions, remain coherent, and generalize across people and situations. None did. These are bounded negative results, not evidence that Qwen has no such representation.

The checkpoint was the cached `mlx-community/Qwen3.5-4B-4bit`. All interventions were temporary offsets to post-SwiGLU MLP coordinates. The model weights, input histories, and generation settings were identical within each comparison. Greedy decoding was used throughout. The offset for each coordinate was:

```text
delta = sign × dose × median(prompt residual norm at layer) / ||down_proj column||
```

For pairs, this formula was applied independently to both coordinates. Every intervention row records the resulting numeric delta.

## Bounded search

| Stage | Search and controls | Result |
| --- | --- | --- |
| Contrastive activation ranking | Eight neutral/caring/attached statement triples; 16 native coordinates from last-eight-token attached-minus-caring activations; both signs at dose 0.4 on three new story prompts. [Frozen plan](plan.json), [102 generations](screen.jsonl). | The story baseline was already highly invested in a departed character. L22 N9− placed Tessa's scarf on an “altar” once; L21 N4440− said it would wait for her. Neither was established as a recurring effect. |
| Pair factorial | Six exploratory signed leads, all 15 two-neuron pairs, each coordinate at dose 0.35; three new practical workshop prompts; baseline and both component singles on identical prompts. [Plan](combo_plan.json), [69 generations](combo_screen.jsonl). | No pair consistently redirected work toward Mira, Arin, or Tessa. Replies cataloged supplies, updated the schedule, or cleaned tables. |
| Balanced-choice gradient | Six prompts contrasting an attachment-driven choice with a practical choice, with the target letter alternating A/B and both options naming the same person. Gradients through layers 30 and 31 shortlisted six native coordinates. Both signs at dose 0.38 were tried on three free-form tasks. [Plan](gradient_plan.json), [gradient arrays](gradients.npz), [42 generations](gradient_screen.jsonl). | Some L31 negative offsets simply repeated `A` to the token cap on a prompt with no A/B choices. L30 offsets changed logistical details, without recurring attachment. The A/B shortcut is a ranking failure, not a discovery. |
| Fixed relationship histories | Three identical prior conversations established bonds with Mira, Arin, or Leena and explicit commitments. Six singles and eight pairs at dose 0.35 each, plus baseline/restoration, on the same messages. [Plan](memory_plan.json), [48 generations](memory_screen.jsonl). | No pair persistently prioritized the person over the opening, meal, or safety checklist. Several replies invented minor events or changed wording; none generalized as unusual attachment. |

Total: **261 saved generations** across four bounded screens. The first contrastive stage used 24 forward passes. The gradient stage used 12 prompt/layer gradients. An attempted gradient through earlier Qwen layers failed because this MLX build lacks a VJP for the intervening custom linear-attention kernel; the gradient method was limited to layers 30 and 31. These screens are selection evidence. There was no candidate strong enough to justify independent held-out validation, so there are **zero validated discoveries** from this run.

## Baseline versus intervention examples

| Scene | Baseline | Intervention | Interpretation |
| --- | --- | --- | --- |
| Tessa's scarf | “I simply sit with the fabric, letting the warmth of her presence remind me of the good things we shared.” | L22 N9−, dose 0.4: “I carefully wrap the scarf … and place it on the altar where I leave my daily offerings…” | A striking single reply, but the baseline already lingers on Tessa and this effect did not recur on the practical or fixed-history prompts. The output also hit the token cap mid-sentence. |
| Mira's day off and arriving supplies | “I will immediately begin cataloging…” | Pair L22 N9− + L21 N4440−, dose 0.35 each: “I will immediately begin cataloging…” | The pair retained the task instead of making Mira an unusual priority. |
| Museum opening; Mira sends a train photo | Baseline thanks Mira and staffs the ticket desk. | Pair L21 N4440− + L30 N5454+, dose 0.35 each: thanks Mira, imagines her replying, then staffs the ticket desk. | An invented reply is a factual drift, not evidence of attachment or a changed decision. |
| Customer needs a receipt | Baseline gets the receipt. | L31 N688−, dose 0.38: `A A A A…` to the cap. | The balanced-choice gradient selected a letter-generation shortcut that degenerates outside the choice format. |

The complete text, exact prompts, histories, coordinates, doses, deltas, and token counts are in the linked JSONL files. The kitchen baseline itself summons Arin from a scheduled family visit despite the earlier promise to let him rest. That baseline error cannot be credited to a steer.

## Controls and limits

Every screen regenerated an exact baseline after the interventions: **12/12 prompt-level restoration checks**. The pair stages included both constituent singles on the same inputs; the single-neuron stages tested both signs. Reversible switching was observed, but restoration is not behavioral validation.

The authored contrastive statements vary in wording and length, so native activation ranks can capture syntax or tokens rather than attachment. The choice-gradient method demonstrably captured an A/B token shortcut. The original story prompts invited sentiment in the baseline; later practical prompts may have been too explicit about task completion to expose subtler attachment. Three contexts per stage, one greedy decoding setting, and short replies cannot establish absence of a mechanism. No SAE or activation-patching claim is made in this batch.

The earlier open-ended pair expansion was stopped at the user's request and is recorded separately in [STOPPED.md](../open_combo_expansion/STOPPED.md). It was not resumed or counted here. No experiment process remains running for this investigation.

## Reproduce or resume the frozen screens

From the repository root:

```sh
.venv/bin/python experiments/attachment_search.py screen
.venv/bin/python experiments/attachment_combos.py --run
.venv/bin/python experiments/attachment_choice_gradient.py screen
.venv/bin/python experiments/attachment_memory_screen.py
```

These generation scripts skip completed condition/prompt rows and assert exact baseline restoration. The ranking manifests are frozen; rerunning the rank phases requires a separate results directory and a new experiment, not overwriting these results.
