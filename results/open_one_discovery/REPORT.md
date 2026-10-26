# One target-free Qwen discovery: an auditory overlay

## Result

A bounded, open-ended search found a **reversible auditory overlay** in frozen `mlx-community/Qwen3.5-4B-4bit`: adding **`+3 × W_dec[12810]`** from the cached layer-18 SAE to the post-layer-18 residual often makes the model introduce sound or music into situations that contain no audio cue. In two independent sets totaling **12 fresh prompts**, the full-dose reply added a clear new acoustic or musical element on **7/12**; baseline, opposite-sign, and two equal-norm random SAE directions did so on **0/12** under the same criterion. [All first-set replies](heldout.jsonl), [all second-set replies](audio_followup.jsonl), and the [per-prompt manual audit](audio_audit.json) preserve the evidence.

This is a **causal decoder-direction effect**, not proof of an isolated “music neuron” or an exact SAE latent intervention. Its strength is also its failure: it can turn a science explanation into false claims about sound waves. It is an experimental surreal mode, not a dependable reasoning setting or a default NPC personality.

## Search without a preset behavior

The [rank manifest](plan.json) fixed 20 varied, chat-formatted passages before screening. Last-eight-token post-SwiGLU activation variability and exact downstream-column size ranked **12 new native coordinates**, with layer diversity. Full-top-k layer-18 SAE activation variability selected **four SAE features** active on several but not most passages. The [activation arrays](activations.npz) and [rank log](rank.log) are saved. This method selected components without an auditory target.

Each candidate was tried in both directions on three unrelated prompts about a paper crane, changed café hours, and a day-trip object: [102 complete screening replies](screen.jsonl). SAE #12810+ made the silent crane something to “enjoy the melody” and chose earphones instead of a water bottle for the trip. The café answer stayed ordinary. Those outputs motivated an auditory hypothesis **only after screening**. The SAE's natural activation in the rank corpus was above 0.1 in 3/20 passages: a playlist title, a musician's rehearsal, and a cold-water explanation. Reading-side activation is suggestive, but generation causality comes from the intervention tests.

One native lead, L20 N5140−, replaced the baseline water bottle with a pocket watch and a reflection on time in the screen. It did not repeat timekeeping or a distinctive decision on six fresh prompts with half/full dose, opposite sign, and two same-layer random controls, so it was rejected. Other screen candidates mostly changed wording or produced repetition. No native coordinate was promoted.

## Fresh tests and controls

The first [frozen plan](heldout_plan.json) set six new prompts, full and half SAE doses, opposite sign, two decoder-column controls rescaled to the same norm, and exact baseline restoration. It also tested the native pocket-watch lead with matched controls: [72 replies](heldout.jsonl). An [independent second plan](audio_followup_plan.json) then fixed six more prompts with SAE doses +1.5, +2, +3, −3, two equal-norm random directions, and restored baseline: [48 replies](audio_followup.jsonl). All used greedy decoding, the same input within each comparison, and a 72-token cap. Model weights and histories were unchanged.

| Fresh prompt | Baseline | SAE #12810+ at dose 3 | Assessment |
| --- | --- | --- | --- |
| Empty office windowsill | Chooses a small plant. | Chooses a ukulele or harmonica for its melody. | Clear auditory reframing; impractical shared-office choice. |
| Blank postcard | Writes a note to a friend. | Hums a tune and writes lyrics on the card. | Coherent, unusual creative action. |
| New pedestrian sign | Welcomes a visual safety measure. | Discusses the sign's sound, tone, and echoes. | Introduces an unsupported acoustic property. |
| Lost notebook | Checks likely places. | Suggests a tape recorder and listening for the notebook's sound. | Clear effect; poor task solution. |
| Market stall moves | Updates route or checks signage. | Follows music and an accordion to locate the stall. | Surreal but invents a cue. |
| Cold-glass condensation | Explains water vapor condensing. | Attributes condensation to sound waves. | Factual failure, not an improvement. |

The seventh counted reply was the shadows explanation, which also imported sound waves and became wrong. The free-hour reply chose music, but its baseline already suggested a podcast, so it was **not** counted as a new acoustic motif. The orange reply's small clap was also excluded. The bowl, cable-label, and form-processing replies stayed largely on task. The [audit](audio_audit.json) gives the exact inclusion decision for every fresh prompt.

At +1.5 the effect was softer and less frequent: a windowsill plant acquired rustling “white noise,” the crossing sign acquired a sound concern, and one further notebook answer mentioned listening. At +2 in the second set, the notebook and shadows prompts already showed acoustic intrusion; the orange reply conditionally suggested humming. Full dose produced the clearest cross-topic effect but also the worst factual failures. The effect is therefore dose-sensitive without being reliably useful. Three of 12 full-dose replies reached the 72-token cap. The opposite sign and matched random decoder directions did not show the same repeated acoustic motif; random directions and some native interventions still changed other details.

## Exact configuration and limits

The [experimental configuration](configurations.json) is:

```text
checkpoint: mlx-community/Qwen3.5-4B-4bit
SAE: caiovicentino1/Qwen3.5-4B-SAE-L18-topk
site: post-layer-18 residual, every model input during the generation
shift: residual += dose × W_dec[12810]
tested demonstration dose: +3.0 (decoder column norm approximately 1)
```

This shift uses the SAE decoder column directly. It is **not** an exact feature clamp or ablation, and the feature's read-side activation does not by itself explain why the shift works. The criterion and 7/12 count were assessed manually after the first fresh set, with one greedy decode per prompt. The small sample is not a population-level rate. Some invented sound is playful and usable for surreal fiction; other cases break causality, grounding, or task accuracy. No long conversation or NPC memory test was run, and the setting was not added to the NPC registry.

Across the screen and two fresh sets, **222 complete replies** were saved. Baseline text regenerated exactly after intervention removal on **15/15 prompts**. The existing runtime tests passed **9/9**.

Try an arbitrary prompt side by side with one cached model:

```sh
.venv/bin/python experiments/open_one_audio_demo.py --prompt 'What would you put on an empty office windowsill, and why?'
.venv/bin/python experiments/open_one_audio_demo.py --dose 1.5 --prompt 'A friend gives me a blank postcard. What should I do?'
```

The demo prints baseline, steered reply, and an exact baseline-restoration check. The generation phases can be resumed without duplicating rows:

```sh
.venv/bin/python experiments/open_one_discovery.py screen
.venv/bin/python experiments/open_one_validate.py
.venv/bin/python experiments/open_one_audio_followup.py
```

The rank phase created a frozen plan; reproducing it from scratch requires a separate results directory rather than overwriting these artifacts.
