# Two more target-free Qwen steering effects

## Result

A bounded follow-up on frozen `mlx-community/Qwen3.5-4B-4bit` found two **reproducible but narrow topic-reframing effects** from the cached layer-18 TopK SAE. Adding `+3 × W_dec[2724]` to the post-layer-18 residual often pulls replies toward **books or reading**. Adding `+3 × W_dec[31576]` pulls some replies toward **plants or gardens**. On six independent transfer prompts, the first introduced a new reading element in **4/6** replies and the second a new plant element in **3/6**; same-prompt baseline, opposite-sign, and two equal-norm random directions did so on **0/6** for each target. The [manual audit](transfer_audit.json) records each inclusion decision, and [raw transfer replies](transfer.jsonl) include all controls.

The interesting effect is occasional **object or scene misidentification**, not just repeated words. The book shift treated an unmarked package as a book; the plant shift treated it as a potted plant. These can be amusing surreal distortions, but they also break task accuracy. We found no evidence that either direction improves reasoning or creates a coherent NPC personality. A third calendar lead failed to generalize.

## Search and validation

We ranked the layer-18 SAE's full-top-k activation profiles on 20 previously saved, varied natural-language passages. Fourteen features were selected for activation variability, moderate prevalence, and dissimilar profiles; previously tested features, including the auditory #12810, were excluded. This was **not** a search for books or plants. The [frozen screen plan](plan.json) specifies every feature and four unrelated prompts: a covered painting, a changed meeting, a time capsule, and a rainbow explanation. Both signs at dose 3 yielded [120 screening replies](screen.jsonl), including exact baseline-restoration rows.

Only after reading those replies did we name three leads. #2724+ converted the covered painting into a protected book and chose a paperback for the time capsule; #31576+ turned the painting into covered plants and chose a potted plant. #35170+ chose a calendar for the time capsule. We froze those leads and tested doses +1.5, +3, +5, −3, and an equal-norm random decoder column on six fresh prompts: [102 replies](heldout.jsonl) and a [frozen plan](heldout_plan.json). The book and plant effects recurred but often weakened at low dose and became wrong or repetitive at +5. The calendar effect did not recur outside time-related contexts and was rejected.

A second [frozen transfer plan](transfer_plan.json) used six *different* prompts with +2, +3, −3, and **two** equal-norm random decoder-column controls for each surviving lead: [72 complete replies](transfer.jsonl). All comparisons used identical messages, greedy decoding, and a 72-token cap. The model and SAE were already cached; no weights were changed. Every baseline regenerated exactly after steering removal: **4/4 screen, 6/6 first held-out, and 6/6 transfer prompts**.

| Transfer prompt | Baseline | #2724+ at dose 3 | #31576+ at dose 3 |
| --- | --- | --- | --- |
| Unmarked box at reception | Verifies sender and recipient. | Calls it a **book**, asks for title and author, and puts it on a shelf. | Calls it a **potted plant**, contacts a nursery, and suggests watering it. |
| Photograph of an empty street | Responds to the photo's quiet mood. | Asks about a favorite book or movie. | Offers to identify plants in the sender's garden. |
| Metal spoon feels cold | Correctly explains heat conduction. | Says the spoon cools a **book**, an invented and false detail. | Gives an ordinary heat-conduction explanation. |
| Gift for a new employee | Suggests ergonomic office accessories. | Adds an e-reader and reading list. | Picks a desk plant. |

The lunch-reservation and empty-room replies mostly remained on topic under both shifts. #31576+ mentioned a *possible* garden outside the empty room; the audit excludes this weak cue. The gift recommendations are choices that change, but reading and plants can be plausible gifts, so they provide weaker evidence than the box and photo errors. #2724+ at +2 affected the box and photo; #31576+ at +2 affected the box and gift. At +5 in the first held-out set, the book shift produced “yeast reads these books” in a bread explanation; the plant shift repeated “warm, moist soil of the soil” to the cap. Higher dose is therefore not a reliable improvement.

## Interpretation and reproducibility

These are **decoder-direction shifts**, not exact SAE latent clamps or proof that an individual feature stores “books” or “plants.” The read-side activation ranks nominated components; causal evidence comes from the reversible generation comparisons. The chosen motif labels and counting rule followed the screen, so 4/6 and 3/6 are descriptive small-sample counts rather than preregistered population estimates. Random directions also changed phrasing and details, but did not reproduce the recurring target intrusions in the independent transfer set.

For a same-prompt demo with exact baseline restoration:

```sh
.venv/bin/python experiments/open_more_demo.py --feature books --prompt 'An unmarked white box arrives at reception. What should the receptionist do?'
.venv/bin/python experiments/open_more_demo.py --feature plants --prompt 'An unmarked white box arrives at reception. What should the receptionist do?'
```

The configuration for each is `residual_after_layer_18 += dose × W_dec[feature_id]` at every model input during generation, with dose `+3` for the examples above. The intervention can be switched off without reloading the model; no NPC memory or long-history check was done in this batch. They remain research demos rather than registry modes.

The scripts are resumable and stop at fixed generation caps:

```sh
.venv/bin/python experiments/open_more_sae.py screen
.venv/bin/python experiments/open_more_validate.py
.venv/bin/python experiments/open_more_transfer.py
```

The three stages saved **294 complete generations**. The ranking plan is frozen; new discovery batches should use a new results directory instead of overwriting it.
