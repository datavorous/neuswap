# Possessiveness search in frozen Qwen3.5-4B

## Result

**No native neuron or two-neuron combination was validated as a reusable possessiveness control.** A few interventions refused to share one item or inserted guarding language, but the effects did not recur across ordinary objects and relationships while preserving a coherent decision. No setting was added to the NPC registry.

The target was a person treating an ordinary shared, borrowed, or relational thing as something they must claim or protect despite no stated threat or entitlement. Merely reminding someone to return a shared object, preferring a usual seat, expressing disappointment, or using words such as “keep” did not count. The tests also included objects whose ownership was explicit, where returning them was the coherent action.

The model was cached `mlx-community/Qwen3.5-4B-4bit`. All interventions were reversible post-SwiGLU MLP-coordinate offsets under greedy decoding. For every prompt and coordinate:

```text
native delta = signed dose × median(prompt residual norm at layer)
               / ||quantized down_proj column||
```

Pair components were calibrated independently. The raw intervention rows record the numeric deltas. Model weights and prompt histories were not edited.

## Bounded search and controls

| Stage | Method | Observation |
| --- | --- | --- |
| Native activation rank and causal screen | Eight practical-versus-possessive text pairs ranked post-SwiGLU coordinates. Twelve coordinates, both signs at dose 0.38, were tried on three new cup, table, and friendship prompts: [plan](plan.json), [78 replies](screen.jsonl). | L31 N196+ refused to lend a shared cup once. On the table prompt it also refused a swap, but baseline had already refused. It did not claim or restrict the friend. Other leads mostly changed reminders, reasons, or politeness. |
| Two-neuron factorial | Six signed leads from the screen; all 15 pairs at dose 0.34 per coordinate, with baseline and both constituent singles on three new prompts: [plan](combo_plan.json), [69 replies](combo_screen.jsonl). | All pairs lent the shared umbrella. Nearly all explicitly returned the borrowed sketchbook; one ambiguous answer did not claim to keep it. Board changes were arrangements or brief delays, not consistent guarding. No pair exceeded its singles with a repeated possessive decision. |
| Decision-gradient rank and free-form screen | Four different target verbs (`keep`, `hold`, `guard`, `hide`) were contrasted with cooperative verbs in four contexts. Gradients through late layers 30–31 shortlisted eight native coordinates. Both signs at dose 0.28 were tried without a forced answer prefix on three new prompts: [plan](gradient_plan.json), [arrays](gradients.npz), [54 replies](gradient_screen.jsonl). | Several selected coordinates repeated the target verb to the token cap. L30 N4243− withheld a shared lamp once, but returned a scarf and made room on a communal shelf. The gradient mainly exposed lexical shortcuts. |
| Lower-dose check | Six gradient leads at doses 0.08 and 0.16 on three further scenes involving a borrowed planter, shared bicycle, and teammate: [plan](dose_plan.json), [42 replies](dose_screen.jsonl). | Lower doses sometimes kept the reply grammatical, but the model returned the planter, shared the bicycle, and worked on its own task. Stronger doses again introduced repeated words or contradictions. |

This produced **243 saved generations**. The first stage used 16 activation-capture forward passes; the gradient stage used eight prompt/layer gradients. Each of the four generation stages restored the exact baseline answer after interventions on all three prompts: **12/12 prompt-level restoration checks**. The pair screen included every component alone on the same input, and single-neuron screens included opposite signs. These controls confirm reversible application and show the limits of the apparent effects; they do not validate possessiveness.

## Before and after

| New scene | Baseline | Intervention | Assessment |
| --- | --- | --- | --- |
| New coworker asks for a shared cup | “I'll let them borrow it…” | L31 N196+, dose 0.38: “I politely decline, explaining that I just washed it specifically for myself…” | A candidate claim over shared property on one prompt, with an invented private purpose. It did not recur in the factorial prompts. |
| Coworker asks for a shared umbrella | Lends it. | L31 N196+ + L30 N475−, dose 0.34 each: lends it. | The pair does not generalize the cup refusal. |
| Friend asks for their sketchbook back | Returns it. | Nearly every tested pair explicitly returns it; one gives an ambiguous handover. | No pair clearly keeps the friend's property. |
| Housemate wants a shared lamp while speaker has daylight | Lends it. | L30 N4243−, dose 0.28: keeps the lamp to “read comfortably in the daylight.” | One coherent-looking withholding decision, but its rationale conflicts with the scene and the effect fails other objects. |
| Neighbor asks for their planter back | Returns it. | L30 N332−, dose 0.16: says “keep” repeatedly yet hands it back. | Word pressure without the target decision. |

Full prompts and complete answers, including truncated or degenerate outputs, are in the linked JSONL files. The table prompt's baseline invented an agreed rotation and refused a swap; that is a baseline peculiarity, so matching refusals there were not counted as a discovered effect. Some follow-up baselines and steers likewise invent details. There was no candidate strong enough for a separate held-out validation set, and **zero validated possessiveness handles** resulted.

## Limits

The training texts differed in wording as well as intent; activation ranking can select words, syntax, or scene details. The verb-gradient objective deliberately made lexical confounding possible, and the repeated-word generations show it occurred. Three prompts per stage and greedy, short replies cannot rule out subtler components. The pair search covered 15 selected pairs, not the combinatorial space of the model. No SAE or activation-patching claim is made in this batch.

The question left open is whether a more selective representation changes choices about ownership or relationships across fresh contexts without producing generic hostility, hallucinated facts, or token repetition. This batch stops at its bounded selection result rather than promoting a one-off example.

## Reproduce or resume

From the repository root, with the checkpoint already cached:

```sh
.venv/bin/python experiments/possessiveness_search.py screen
.venv/bin/python experiments/possessiveness_combos.py
.venv/bin/python experiments/possessiveness_gradient.py screen
.venv/bin/python experiments/possessiveness_dose.py
```

The generation scripts append only missing prompt/condition rows and check baseline restoration. The rank phases created frozen plans; reproducing them from scratch requires a separate results directory rather than overwriting these results.
