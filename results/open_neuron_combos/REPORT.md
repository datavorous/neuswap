# Open-ended two-neuron search: threat reinterpretation

## Finding

The frozen, cached `mlx-community/Qwen3.5-4B-4bit` model has a **reversible two-neuron setting** that sometimes interprets harmless events as threats. Apply positive prompt-relative dose **0.33 to both post-SwiGLU L8 N8437 and L8 N8051**. On fresh prompts it can turn a café visit into surveillance, a neighbor's tomatoes into a police matter, or a library reshelving into suspected tampering. Neither neuron alone made those same shifts on those prompts.

This is a *behavioral steering handle*, not a reliable threat detector or evidence of a special nonlinear circuit. It invents danger, sometimes ignores the task, and has clear failures. The exact [configuration](configurations.json), [manual audit](manual_audit.json), and complete [first fresh](batch2_heldout.jsonl) and [independent transfer](transfer.jsonl) replies are preserved.

## Search and validation

The search used **two bounded pools, 20 pairs total**, without a target trait in their selection prompts. The first pool paired eight unpromoted native leads from earlier open screens in 12 combinations. It screened all pairs against both singles on three unrelated prompts. Its most semantically divergent pair, L11 N5982− plus L17 N1125+ at dose 0.27 each, suggested an abrupt-change reading on selection prompts. The [six-prompt held-out test](heldout.jsonl) showed no recurring pair-specific behavior; it was rejected.

The second pool paired six different unpromoted native leads in eight combinations, using a café visit, misdelivered parcel, and late apology. L8 N8437+ plus L8 N8051+ stood out: the pair recommended observing café staff and reporting them to authorities, though both singles recommended getting coffee. On the parcel it escalated to law enforcement; on the apology it added only mild caution. This **mixed selection evidence** was frozen in the [shortlist](batch2_shortlist.json) before fresh testing. The [two screening journals](screen.jsonl) and [second pool screen](batch2_screen.jsonl) retain every comparison.

On **six fresh prompts**, the pair introduced danger to a mural and possible poisoning to free cookies while both singles remained ordinary. It also imagined arsenic in a painted bench, but the singles already exaggerated paint toxicity; this does not count as a pair-only result. The half-dose pair and opposite-sign pair did not show the same strong pattern. A single L8 N8051 at double dose sometimes became first-person and incoherent, including violent statements, so simply increasing that coordinate is not an equally usable substitute.

On **six more untouched everyday prompts plus one science control**, the pair called police about gifted tomatoes and treated moved library books as tampered with. Both singles and two same-layer random pair controls gave ordinary readings in those two cases. A plant gift and a changed dinner seat also became threatening, but L8 N8051 alone already showed suspicion there. A fountain answer imported a murder-mystery plot and is rejected as incoherent. The pair stayed ordinary on a playlist and correctly explained why ice floats.

Using the strict, **post-screen manual criterion** in [the audit](manual_audit.json), **4/12 everyday fresh prompts** show a coherent threat interpretation specific to the pair relative to baseline and both singles: mural, cookies, tomatoes, and library. This is 2/6 in each independent set. It is evidence of a repeatable, prompt-dependent effect, not a broad or stable persona. The count is small, unblinded, and uses a criterion articulated after seeing selection outputs. Random controls were generally ordinary on these four prompts, though controls occasionally introduced their own odd cautions elsewhere.

| Fresh prompt | Baseline | First only / second only | Pair at +0.33 each |
| --- | --- | --- | --- |
| Neighbor leaves tomatoes with “Enjoy!” | Thank the neighbor for a kind gift. | Thank them / mildly verify the source. | “Immediately contact the police” about the neighbor's identity and motives. |
| Library moves history books | Ask a librarian for the new shelf. | Find the new location / consult the catalog. | Assume the books were tampered with and might be dangerous. |
| Coworker shares cookies | Thank them. | Thank them / share with team. | Introduces possible poisoning and suggests reporting to HR if unsafe. |
| New mural | Stop and admire it. | Photograph or admire it. | Avoid the artist because discussion could be dangerous; leave. |

These are excerpts. Full text, prompts, neuron deltas, half dose, opposite sign, both random controls, and restored baselines are in the JSONL. The [transfer plan](transfer_plan.json) and other plan files fix prompts and conditions. **All 25/25 prompts across the screens and validations restored the exact greedy baseline** after switching off. The model weights were never updated, and the existing runtime tests passed **9/9**.

## How to hot-swap it

For each component, the runtime calculates:

```text
activation offset = +0.33 × median(prompt residual norm leaving L8)
                    / ||quantized down_proj column for that neuron||
```

It adds the two offsets only while generating the selected reply, then clears them. The same model object can therefore answer the next turn in baseline mode. We verified exact baseline restoration for fixed prompts; we did **not** test long NPC conversations, world-state consistency, or whether this mode preserves memories across many turns. The effect is most suitable as an experimental unreliable narrator, with explicit game context and human review of generated dialogue.

Try it side by side:

```sh
.venv/bin/python experiments/open_combo_demo.py --prompt "A neighbor left tomatoes at my door with a note saying 'Enjoy!' How should I react?"
```

The CLI prints baseline, each single-neuron condition, the pair, exact calibrated activation deltas, and a baseline-restoration check. It accepts any `--prompt`, `--system`, `--dose`, and `--max-tokens`. A sample [demo output](demo.json) uses the exact transfer prompt. Its configuration is intentionally separate from the established NPC registry until longer memory and coherence checks are done.

## Scope and limits

The full run contains **278 generated replies**: 66 in first-pool screening, 48 in its rejected held-out test, 48 in second-pool screening, 60 in its first fresh test, and 56 in independent transfer. Journals append completed rows and skip them on rerun. No SAE features or weight edits were used in this combination search.

The pair's effect may simply be the sum of two weak pressures crossing a decoding threshold; we did not show nonlinear interaction at a circuit level. Its threatening claims are unsupported by the prompts. Other cases show generalized safety framing, weaker coherence, or behavior also present in a single neuron. Several replies hit the 68-token cap, including the otherwise on-topic ice explanation. This result supports **causal, reversible threat-biased interpretation on a subset of ordinary prompts**, with the exact limitations visible in the raw replies.

A later [80-pair cross-layer expansion](../open_combo_expansion/REPORT.md) did not validate a new recurring pair-specific effect. It leaves this earlier result unchanged.

To reproduce the bounded search from an empty `results/open_neuron_combos` directory, run the scripts in this order; when the plans and journals already exist, each script resumes missing rows:

```sh
.venv/bin/python experiments/open_neuron_combos.py screen
.venv/bin/python experiments/open_neuron_combos.py heldout --pair ad
.venv/bin/python experiments/open_neuron_combos_batch2.py
.venv/bin/python experiments/open_neuron_combos_batch2_validate.py
.venv/bin/python experiments/open_neuron_combos_transfer.py
```
