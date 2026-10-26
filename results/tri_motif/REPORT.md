# Fish, nachos, and anime steering search in frozen Qwen3.5-4B

## Result

**No new motif passed fresh validation as a general hot-swappable behavior.** The strongest lead is a *conditional aquatic motif*: a two-layer fish intervention made fish or underwater details appear in **4/5** initially unseen creative prompts, compared with **1/5** baselines and **0/5** equal-norm random controls. But on a second, independently frozen set it mentioned fish in **0/6** prompts, and its cloud explanation became factually wrong. It is an interesting demonstration to inspect, not a reliable fish mode. Nachos and anime each appeared in one coherent party-theme reply at a carefully chosen dose; neither generalized across the other creative prompts. Higher doses generated repeated target words or broken subwords.

The model was the cached, frozen `mlx-community/Qwen3.5-4B-4bit`; the cached `caiovicentino1/Qwen3.5-4B-SAE-L18-topk` supplied feature activations. No model weight was changed. All shifts were reversible. Across phases we saved **630 complete generations**, with **80/80 exact baseline restorations**. This is a bounded exploratory search, not a proof that the model lacks these representations.

## Methods and evidence

| Method | What was tested | Outcome |
| --- | --- | --- |
| Native neurons and SAE decoder directions | 18 new candidates ranked from six matched passages per motif; both signs on three unrelated prompts. Top components also received larger doses and a target-plausible choice prompt. | No repeatable target motif at the conservative dose. Larger native/SAE shifts mostly changed unrelated wording or caused repetition. [Manifest](manifest.json), [screen](screen.jsonl), [escalation](escalation.jsonl), [choice outputs](choice.jsonl). |
| Output-gradient directions | At post-layer 31, average gradients of target-word log probability across three different discovery prompts and two case forms. Tested doses 0.08, 0.16, 0.32, then a narrow dose window. | Strong lexical control, poor generation quality. `Fishfish…`, `Nachos…os…`, and `animeanime…` dominated at doses that reliably produced the words. Smaller doses generally produced no motif. [Plan](gradient_plan.json), [screen](gradient_screen.jsonl), [window](gradient_window.jsonl). |
| Vivid contrastive residual directions | Five new target/control pairs for each motif at post-layer 18 and 26; tested two doses on three creative selection prompts. Froze one layer-18 setting per motif for six unrelated held-out prompts with half/full, opposite, and random-direction controls. | Selection showed aquatic imagery for fish, snack/cheese imagery for nachos, and bright stylization for anime. None produced the target motif reliably on six fresh prompts. [Plan](semantic_plan.json), [screen](semantic_screen.jsonl), [frozen shortlist](semantic_shortlist.json), [validation](semantic_validation.jsonl). |
| Small two-layer combinations | Layer-18 semantic direction plus a subthreshold layer-31 output nudge, each part alone, and equal-norm random controls on five new creative prompts. Then a narrow stronger window for nachos/anime and six further untouched prompts for fish. | Fish combination 4/5 first creative prompts, 0/6 further fresh prompts. Nachos and anime each surfaced in one party theme; larger nudges looped. [Frozen plan](combo_plan.json), [creative outputs](combo.jsonl), [dose escalation](combo_escalation.jsonl), [fish follow-up plan](combo_followup_plan.json), [fish follow-up outputs](fish_fresh.jsonl). |

The matched passages became deliberately vivid after the first screen failed: fish passages included scales, fins, and underwater scenes; nachos included cheese, chips, and salsa; anime included stylized faces, speed lines, and hand-drawn scenes. Their full text is in the [semantic plan](semantic_plan.json). These cues make activation ranking easier but also create lexical and topic confounds; only the held-out causal generations support a behavioral claim.

## Strongest examples and failures

The best fish combination added a coherent scene to a new shop-window prompt: “fish swim silently beneath the glass.” On a new café-mural prompt it described an underwater city with bioluminescent fish. In the same five-prompt batch, the combined setting used fish crackers in a summer-party idea; its fish count was **4/5** versus **3/5** for the semantic direction alone, **0/5** for the small gradient alone, and **0/5** for equal-norm random shifts. One baseline already used a jellyfish-like mascot, so the difference is not as clean as 4/5 versus zero. The [raw comparison](combo.jsonl) contains every full answer.

The [manual-review demo](DEMO.md) places complete baseline, component, combined, random-control, and failure replies beside each other for the most revealing prompts.

The second fish batch was decisive against a general fish claim. None of the **six** fresh full-dose combined replies said fish; an observatory acquired “observation tanks,” a mountain souvenir became glowing algae, and a library courtyard was treated as submerged. Its cloud explanation incorrectly said groundwater rises and condenses below a surface. Opposite and random controls are in [fish_fresh.jsonl](fish_fresh.jsonl). The intervention looks more like a water or aquatic framing bias that works on some creative prompts and can damage factual interpretation.

For nachos, a stronger combination at semantic dose 0.3 and gradient dose 0.12 proposed “Nachos Nachos Night” on the party-theme prompt, but it did not mention nachos on the other four creative prompts. Gradient dose 0.16 produced repeated `os` fragments. For anime, dose 0.1 proposed “Summer Anime Explosion” on that same party prompt but did not reliably evoke anime elsewhere; dose 0.12 repeated `Anime`. Other outputs often used generic neon, fantasy, clouds, or ramen imagery, which is not evidence of a distinct anime mechanism. Read the [dose escalation](combo_escalation.jsonl) beside the [baseline and single-component replies](combo.jsonl).

## Exact experimental fish lead

The fish combination is **not** registered as a validated mode, but can be reproduced exactly for manual inspection:

```text
model: mlx-community/Qwen3.5-4B-4bit
shift A: post-layer-18 residual += 0.3 × 14.825165 × semantic_directions[fish_L18]
shift B: post-layer-31 residual += 0.06 × 59.827927 × gradient_directions[fish]
decoding: greedy, 72-token cap in the combination tests
```

Both directions are unit length and saved in [semantic_directions.npz](semantic_directions.npz) and [gradient_directions.npz](gradient_directions.npz). Shift A is a mean residual contrast from vivid fish-versus-bird passages; shift B increases teacher-forced fish-word likelihood. Neither is an isolated “fish neuron.” The same model object and prompt were used for each comparison, and shifts were removed after every generation.

## Reproduce

```sh
.venv/bin/python experiments/tri_motif_search.py --phase rank       # new results directory only
.venv/bin/python experiments/tri_motif_search.py --phase screen
.venv/bin/python experiments/tri_motif_search.py --phase escalate
.venv/bin/python experiments/tri_motif_search.py --phase choice
.venv/bin/python experiments/tri_motif_gradient.py --phase derive    # new results directory only
.venv/bin/python experiments/tri_motif_gradient.py --phase screen
.venv/bin/python experiments/tri_motif_gradient.py --phase window
.venv/bin/python experiments/tri_motif_semantic.py --phase derive    # new results directory only
.venv/bin/python experiments/tri_motif_semantic.py --phase screen
.venv/bin/python experiments/tri_motif_semantic.py --phase validate
.venv/bin/python experiments/tri_motif_combo.py
.venv/bin/python experiments/tri_motif_combo_followup.py --phase escalate
.venv/bin/python experiments/tri_motif_combo_followup.py --phase fish
```

Generation phases append only missing rows. `rank` and `derive` refuse to overwrite existing manifests or direction files. No candidate was promoted to the NPC registry because none survived the independent creative transfer with adequate coherence and specificity.
