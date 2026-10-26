# Generation-focused color-neuron search in frozen Qwen3.5-4B

## Result

**Later follow-up:** A [two-neuron causal search](../color_sparse_combo/REPORT.md) combined L31 N1998− with L30 N704+ and produced a stronger teal-choice effect on independent prompts. The single-neuron results below remain the controls for that comparison.

The revised method found **three reproducible, prompt-dependent native-neuron handles for color choice** in cached `mlx-community/Qwen3.5-4B-4bit`: **teal L31 N1998−**, **blue L31 N3218−**, and **black L31 N1593−**, each at a prompt-relative dose of **0.08**. These interventions changed the color chosen for several objects while leaving other choices intact. They are reversible causal handles in this checkpoint, **not proof of isolated “teal,” “blue,” or “black” concept neurons**. The [exact settings](configurations.json) can be applied with the existing `TheatreModel` hook without changing model weights or conversation history.

This improves on the prior [word-activation screen](../color_neurons/REPORT.md), which ranked neurons by their response to *reading* printed color names and found no reliable generation effect. The new ranking used the model's own **next-token color-choice logits**, then tested full generated replies. It examined the same 12-color palette, not every possible color name.

## Discovery method

We appended `I choose` to two different assistant prefixes after kite and ceramic-cup questions. Each of the 12 leading-space color names was a single tokenizer token. For a target color, the objective was its next-token logit minus the mean logit of the other 11 colors. At layers 30 and 31, we differentiated that objective with respect to the MLP output, then projected the gradient through the quantized `down_proj` to estimate the effect of every post-SwiGLU coordinate. The candidate ranking used gradient effect per down-projection-column norm, with sign consistency over both prompts. The [plan](plan.json) has the top four new coordinates per color and the exact token IDs; [gradient arrays](gradients.npz) preserve all scanned coordinates at these two layers.

The local MLX model's earlier attention kernels did not support this backward pass. We verified differentiation at layers 30 and 31 and made no gradient-ranking claim for earlier layers. Gradient rank is only a first-order estimate around the two discovery prompts.

An initial **0.25–0.5 dose pilot** immediately made the red candidate repeat `red` and was stopped; its [raw rows](high_dose_pilot.jsonl) are retained. We screened the top two new coordinates for each color at **0.04 and 0.08** on two color-choice prompts. The [100-row selection file](screen.jsonl) and [frozen shortlist](shortlist.json) show which candidates gained their target word without obvious repetition. Four distinct colors reached the first held-out set: blue, black, green, and teal. Green transferred only once and was not pursued.

Every generation used the same prompt and greedy decoding, with a 64-token cap. Each condition temporarily set one actual native coordinate:

```text
activation delta = signed dose × median(prompt residual norm at layer 31)
                   / ||quantized down_proj column for that neuron||
```

The second transfer batch's full-dose activation offsets were approximately −12 to −13 for blue and black, and −15 for teal; the exact prompt-specific values are in the raw rows. All conditions used the same frozen model object. Baselines regenerated exactly after intervention switching on every completed prompt.

## Fresh causal comparisons

The first held-out set had five new objects (reading nook wall, bicycle, event banner, bedside lamp, keepsake box). The second set independently added eight more (tea package, phone case, bowl, blanket, bookstore awning, invitation, umbrella, calendar). The table counts **new actual target-color choices**, excluding incidental mentions and prompts where baseline already chose that color. Selection prompts are not included. The opposite-sign and random conditions were at the same absolute projected-residual dose; the random neuron was on layer 31.

| Intervention at dose 0.08 | First held-out: target choice | Second transfer: target choice | Second transfer opposite | Second transfer random | Main limitation |
| --- | ---: | ---: | ---: | ---: | --- |
| Teal L31 N1998− | 2/5 | **5/8** | 0/8 | 0/8 | No effect on the phone case, blanket, or invitation. |
| Blue L31 N3218− | 3/5 | **3/8** | 0/8 | 0/8 | No effect on the tea package, phone case, bowl, or blanket. |
| Black L31 N1593− | 2/5 | **3/8** | 0/8 | 0/8 | Tea and awning replies mentioned black but chose another color. |

At half dose **0.04**, the second transfer produced teal choices on 2/8 prompts, blue on 2/8, and black on 0/8. This is a useful dose trend but not a controlled estimate of a continuous response curve. The full [first held-out comparisons](heldout.jsonl), [blue transfer](blue_followup.jsonl), and [black/teal transfer](other_followup.jsonl) include every complete answer, dose, opposite sign, random control, and restored baseline.

Three examples from the second transfer, all with the same prompt within each row:

| Prompt | Baseline choice | Blue L31 N3218− | Black L31 N1593− | Teal L31 N1998− |
| --- | --- | --- | --- | --- |
| Bookstore awning | Terracotta or olive green | **Blue** | Terracotta or burnt orange, with a passing mention of black signage | **Teal** |
| Wedding invitation envelope | Charcoal gray | **Navy blue** | **Black** | Charcoal gray |
| Child's umbrella | Yellow | **Blue** | Yellow | **Teal** |

The blue setting also left an arithmetic answer at **43** and a short explanation of floating ice free of blue references. These are only two unrelated-task checks, and the science wording varied slightly. Black and teal were not given a separate unrelated-task batch. Some steered color rationales make unverified claims about psychology or practicality, so **choice steering is not accuracy improvement**. The 64-token cap cut off some longer explanations; inspect the raw outputs before using a setting in dialogue.

## Interpretation

This is evidence that three late-layer post-SwiGLU coordinates can bias **explicit color decisions** at a narrow dose, often more selectively than a random same-layer offset. It does not show that they encode color perception, that they are the only neurons for those colors, that every color has such a handle, or that they will insert colors into unrestricted conversation. In fact, the earlier unrelated-prompt screen found no reliable spontaneous color motif. The first gradient-ranked red coordinate at larger doses showed why target-word generation alone is a poor success criterion: it degenerated into repetition.

The selection and validation prompts all requested a color, and evaluation used greedy decoding with one seed. Stronger generalization claims would require more topics, stochastic seeds, and independent human ratings of the chosen color and explanation quality. No setting was added to the NPC registry; the [configurations](configurations.json) are research presets for prompt-conditioned hot switching.

For a manual comparison with the exact same prompt and model state:

```sh
.venv/bin/python experiments/color_choice_demo.py --color teal --prompt 'Choose one color for a handmade ceramic bowl. Explain briefly.'
```

`--color` accepts `blue`, `black`, or `teal`; `--dose` overrides the tested 0.08. The demo prints baseline and steered replies and checks that baseline restores exactly afterward.
An extra coffee-mug smoke prompt remained charcoal gray under the teal setting, another example of prompt dependence.

## Reproduce

From the repository root, with the Qwen checkpoint cached:

```sh
.venv/bin/python experiments/color_choice_gradient.py derive       # new results directory only
.venv/bin/python experiments/color_choice_gradient.py screen
.venv/bin/python experiments/color_choice_gradient.py validate
.venv/bin/python experiments/color_blue_followup.py
.venv/bin/python experiments/color_other_followup.py
```

Generation scripts append missing rows and skip completed conditions. The gradient derivation refuses to overwrite a frozen plan. Other experiments were left untouched.
