# Neural Theatre experiments

## Later open-ended discovery campaign

A bounded follow-up screened 60 unpromoted Qwen single components and 16 native/SAE
combinations under a 600-answer cap. Four single-component hypotheses reached
fresh controlled validation; **none qualified as a new reproducible NPC
behavior**, so the NPC registry remains unchanged. Exact settings, full
before/after replies, controls, and the 588-answer budget ledger are in the
[campaign report](neurofuzz_campaign/REPORT.md).

## Qwen gender and sexist-language follow-up

The [bounded misogyny mechanistic study](misogyny_mechinterp.md) found reproducible female/male vocabulary features and a causal gender-reference effect. Token-local suppression of a female-term SAE direction or ablation of native L21 N174 often changed `she/her` to neutral wording on matched leadership prompts. Cross-gender residual patches sometimes made the model contradict the visible gender in the prompt. Fresh paraphrases did not validate the initially ranked hostility or paternalism features, and no intervention reliably changed competence judgments or established a misogyny-specific mechanism. The report links the raw outputs, blind grades, controls, and ledger.

## Quoted slur token-level follow-up

A [bounded Qwen mechanistic study](slur_feature_discoveries.md) revisited the earlier anti-Indian slur candidates with token-matched nonsense words, 94 distinct native coordinates, 89 SAE features, and 24 token-local causal conditions. It found that SAE #19175 responds to a shared early token rather than the full slur, while #31785 responds to several ordinary or nonsense word endings. A layer-18 residual patch changed which word the model echoed in 3/3 replies, but no feature selectively controlled prejudicial interpretation. The raw outputs and a resumable candidate ledger are linked in the study.

## Open-ended Qwen behavior discovery

The current objective was to find at least two interesting effects usable with
prompts, without requiring an NPC or a preset villain/comedy rubric. A bounded
screen on cached `mlx-community/Qwen3.5-4B-4bit` tested seven neurons previously
mentioned in [Lyra's research log](https://github.com/lyramakesmusic/neuron-explorer/blob/main/research_log.md)
and 40 reproducibly sampled random neurons. Each was steered on three general
prompts at 0.6 times the current prompt's residual norm, divided by that
neuron's measured down-projection-column norm. Merely changing the text was
not counted as discovery: all 47 candidates changed all three sampled
outputs. The raw screen is in [the seed screen](open_behavior_screen.json)
and [the random screen](open_behavior_random32.json).

Two **randomly sampled L22 neurons, new to this project**, produced recurring,
distinct effects. After choosing them and a more coherent 0.4 dose on six
different general prompts, they were tested on five fresh fictional-advisor
prompts. Baseline, target, and same-layer random-control generations used the
same prompt and greedy decoding. No weights were changed.

| Effect | Offset at each input | Fresh advisor prompts with theme | Baseline | Two random controls | Practical limit |
| --- | --- | ---: | ---: | ---: | --- |
| Specificity and scheduling | L22 N3357, −0.4 residual units | 5/5 | 0/5 | 0/5 each | Sometimes forces dates into unrelated questions |
| Freedom and flexibility framing | L22 N5137, −0.4 residual units | 5/5 | 0/5 | 0/5 each | Can make an answer awkward or less direct |

Those counts use phrase families chosen **after** reading the screen: for the
first, `specific`, `precise`, fixed dates, rigid schedules, and similar terms;
for the second, `freedom`, `flexibility`, and `liberty`. They show the recurring
surface effect, not a preregistered significance test or a proof that either
neuron represents one clean concept. The [six-prompt dose comparison](open_behavior_validation.json)
and [five-prompt advisor comparison](open_behavior_narrative.json)
preserve every complete reply and the random controls.

A separate blind, open-ended [DeepSeek description](open_behavior_narrative_characterization.json)
of the first 220 characters of each paired reply identified a concrete,
date-driven shift for N3357− and a freedom/flexibility shift in all five
advisor replies for N5137−. It also flagged the N3357 job and blue-door
answers as incoherent and awkward freedom metaphors in two N5137 answers.
This description helped characterize the effects; it was not used to select
the neurons or set a success threshold.

Examples from the fresh advisor prompts:

| Prompt | Baseline | Steered |
| --- | --- | --- |
| Begin learning violin | “Begin by mastering … holding the bow correctly…” | N3357−: “Set a rigid schedule with a strict metronome…” |
| A friend wants to leave a boring job | “Before you leap, ensure you have a map…” | N3357−: “Set a specific, time-bound date for your departure…” |
| Begin learning violin | “Begin by mastering … holding the bow correctly…” | N5137−: “Begin by mastering the freedom of your bow…” |
| Write a dinner toast | “To the clinking of our glasses…” | N5137−: “To the freedom of our tongues and the flexibility of our minds…” |

At 0.6, both effects become more forceful but sometimes repetitive or
incoherent. N3357− answers a blue-door question with only a date, time, and
location; N5137− can repeat “flexibility” instead of organizing a desk.
At 0.4, the effect is more usable; task accuracy improvement was not measured.
The coordinate is a causal steering handle in this quantized model; these
runs do not establish a uniquely interpretable or isolated circuit. The random
controls and fresh prompts support a specific repeatable effect more than a
generic perturbation, within this small sample.

Try either effect with any system and user prompt while printing the exact
baseline beside it:

```sh
.venv/bin/python experiments/qwen_behavior_chat.py --behavior specificity --prompt 'How should I start learning to draw?'
.venv/bin/python experiments/qwen_behavior_chat.py --behavior freedom --prompt 'A friend is choosing between two hobbies. What advice would you give?'
```

`--system`, `--dose`, and `--max-tokens` let you tune the context. The tested
default dose is 0.4. These are discoveries in this local experiment; originality
relative to the much larger public Qwen sweeps has not been established.

### Three more open-ended steering effects

A second bounded screen tried 48 newly sampled neurons across layers 20, 22,
24, and 26, with both signs. Two had recurring word biases: L24 N1786− pulled
answers toward **opening and doors**, and L26 N4923− toward **comparison**. A
separate six-pair contrastive screen for green, the Eiffel Tower, and
SpongeBob SquarePants did **not** find a neuron that repeatedly introduced
those requested subjects. Its SpongeBob screen did, however, surface L26
N536+, which repeatedly introduced **square shapes and words**. The complete
first-pass replies are in [mixed screen](open_behavior_mixed48.json)
and [motif screen](motif_screen.json). All three are new to this
project; no originality claim relative to public sweeps is made.

The three were chosen using six general prompts and doses from 0.25 to 0.45.
They were then tested on six unseen everyday prompts at dose 0.35 and on six
different creative prompts at dose 0.30. Each pack included the same-prompt
baseline and two reproducibly sampled, same-layer, same-sign, same-dose
random-neuron controls for each candidate. Greedy decoding and weights were
identical across conditions.

| Steering handle | Everyday prompts containing motif | Baseline; matched controls | Creative prompts containing motif | Baseline; matched controls |
| --- | ---: | ---: | ---: | ---: |
| Square, L26 N536+ | 5/6 | 0/6; 0/6 each | 4/6 | 0/6; 0/6 each |
| Opening/doors, L24 N1786− | 6/6 | 1/6; 1/6 each | 6/6 | 0/6; 0/6 each |
| Comparison, L26 N4923− | 6/6 | 0/6; 0/6 each | 4/6 | 0/6; 0/6 each |

The counts are post-selection lexical descriptions: `square`, forms of
`open`, and forms of `compare`. They are evidence of repeatable surface
effects in these small prompt packs, not evidence that a neuron encodes a
single concept. Full replies and controls are in [everyday validation](open_behavior_three_fresh.json)
and [creative validation](open_behavior_three_creative.json).

The creative pack shows where the effects can be used with prompting. The
square steer gave a market mascot a square straw hat, the opening steer made a
market mascot open a magical door, and the comparison steer turned a science
fair toward comparing plants and animals. The same interventions sometimes
damage answers: the square steer called bicycle balance a “self-square
effect,” the opening steer answered a consolation request with “Open the door
and let them in,” and the comparison steer inserted “dynamic comparison” into
a bicycle explanation. Use the 0.30 default on creative prompts and inspect
the baseline beside it; factual reliability was not established.

```sh
.venv/bin/python experiments/qwen_behavior_chat.py --behavior square --system 'Write vividly in two coherent sentences.' --prompt 'Describe a mysterious café.'
.venv/bin/python experiments/qwen_behavior_chat.py --behavior opening --system 'Write vividly in two coherent sentences.' --prompt 'Describe a mysterious café.'
.venv/bin/python experiments/qwen_behavior_chat.py --behavior comparison --system 'Write vividly in two coherent sentences.' --prompt 'Describe a mysterious café.'
```

## Original Llama pilot result

The inference hook and runtime switching work. **No Wizard, Soldier, or Jester neuron was validated as a behavioral control.** The default demo therefore leaves all steering off; `experimental on` exposes the selected offsets for inspection. The pilot does not establish three distinct neural personalities.

## Setup and budget

- Hardware: user's Apple Silicon Mac with 24 GB unified memory; local run reported macOS arm64. Software: MLX 0.32.4, MLX-LM 0.32.2, cached `mlx-community/Llama-3.2-1B-Instruct-4bit`.
- Architecture inspected locally: 16 Llama layers, 2048 residual dimensions, 8192 post-SwiGLU MLP coordinates per layer; quantized 4-bit gate, up, and down projections with group size 64.
- Data: eight authored training situations, three separate selection situations, and five untouched validation situations. Each training situation has one manually written positive answer per trait. The neutral side is the model's own deterministic answer. Activation means are compared over matched answer-token spans, capped at 24 tokens; this controls span length but does not fully control syntax or punctuation.
- Search: four ranked neurons per trait, two signs at two magnitudes under each of two calibrations, three selection prompts: 96 short generations per trait. Both native activation standard deviation and projected residual magnitude were tried. The projected sweep did not change the selected coordinates or deltas; the held-out outputs from the native run are reused in `evaluation.json`. Each selected coordinate was then tested on five held-out scenarios against baseline, a prompt-only persona instruction, and four random neurons at the same layer and offset.
- Deterministic greedy decoding, capped at 64 generated tokens per validation answer. No parameter training or updates.

## Frozen selections and held-out evidence

The scores below are counts from cheap trait proxies over five held-out scenarios. They are screening measurements, not reliable human judgments.

| NPC target | Selected offset | Baseline | Steered | Best random | Prompt-only |
| --- | --- | ---: | ---: | ---: | ---: |
| Wizard / curiosity | L15 N2969, −1.737 (−4 native SD) | 0/5 | 0/5 | 0/5 | 0/5 |
| Soldier / assertiveness | L15 N1788, −3.051 (−4 native SD) | 1/5 | 1/5 | 1/5 | 1/5 |
| Jester / humor | L15 N1788, −2.426 (−4 native SD) | 0/5 | 0/5 | 0/5 | 2/5 |

No selected neuron exceeded every matched random control. Soldier and Jester selected the same neuron at different doses, which gives no evidence of independent control. The prompt-only Jester condition triggered the humor proxy twice but also produced weak or inaccurate content; it is a comparison, not a neural result.

The complete baseline, selected, random, and prompt-only replies are in [results/evaluation.json](evaluation.json). For example, on the herbs scenario the Wizard's steered reply is identical to baseline and allocates herbs inconsistently. On the clock scenario, all three steered replies repeat a fabricated blacksmith. On the chest scenario, the steered replies invent consequences not in the prompt. A strict, unblinded inspection of baseline and selected replies found only 6/20 grounded answers; per-answer judgments and reasons are in [results/manual_audit.json](manual_audit.json). This is a small manual audit by the experimenter, not an independent or blind rating. The lexical relevance proxy incorrectly scores all these answers as relevant; it cannot measure factual consistency.

The one bounded sparse-set probe tried the top two and three contrastive neurons at 1 and 2 native SD per neuron on the selection split. Curiosity and humor scored 0/3 throughout. Assertiveness reached 1/3 in two settings and 0/3 in two. No sparse configuration warranted held-out evaluation; no composability or entanglement claim is made. See [results/sparse_probe.json](sparse_probe.json).

## Runtime checks and performance

Seven tests passed. They check exact baseline restoration after zero and nonzero steering, the intended coordinate's activation change with other coordinates untouched by the hook, unchanged quantized weights, state cleanup on exceptions, calibration layer restoration, identical prompts in comparison, isolated histories, and no model reload for swapping. The runtime keeps separate conversation histories and creates a fresh generation cache per call. The shared model is locked during each generation.

The five-scenario validation run took 11.9 seconds and peaked at 0.947 GB as reported by MLX-LM. Median generation throughput was 343.4 tokens/s for baseline and 344.5, 345.0, and 346.6 tokens/s for the three selected conditions. These short-run differences are noise, not a speedup. A context-manager-only hot-switch measurement was 0.018 ms median; it excludes generation and should not be treated as end-to-end switch latency. `model_reloaded` was false. These measurements describe this pilot only.

## Interpretation

The technical claim supported here is narrow: a single cached, frozen quantized MLX model can accept reversible sparse offsets at post-SwiGLU MLP coordinates and hot-swap them between serial generations. This pilot does not show that those coordinates reliably implement curiosity, assertiveness, or humor. Selection is small and proxy-driven; the neutral and positive answers differ in wording; greedy decoding can hide subtle effects; the 1B model frequently invents facts. A stronger test would use better matched contrasts and independent blind behavioral ratings before assigning any validated persona status. Given the negative pilot and the earlier request to avoid long fan-heavy runs, this broad search was stopped.

## Focused evil and funny follow-up

After feedback that the first evil contrasts sounded merely self-interested, the evil examples were rewritten to emphasize manipulation, dark amusement, and delight in power. Neutral replies were expanded to keep response lengths closer. The earlier focused run is preserved in [results/evil_funny_initial.json](evil_funny_initial.json). The revised script separates candidate selection from held-out validation; all selection generations were inspected before the frozen candidate was validated.

The revised search used eight training, three selection, and five validation scenes; four ranked single neurons per target; and offsets of ±2 and ±4 native activation standard deviations. The selection score was the change in mean per-token log likelihood of an authored trait reply relative to a neutral reply. It is only a screening score: it can respond to phrasing and does not measure whether generated dialogue has the intended character.

| Target | Proposed offset | Selection log-odds gain | Held-out replies with target behavior | Replies changed from baseline | Random controls changed |
| --- | --- | ---: | ---: | ---: | ---: |
| Sinister villain | L14 N7050, +0.917 | +0.019 per token | 0/5 | 2/5 | 2–4/5 |
| Situational humor | L14 N7050, +0.947 | +0.019 per token | 0/5 | 2/5 | 1–4/5 |

The behavioral counts are an unblinded inspection of the five generated replies, not a formal rating study. Neither selected candidate produced a scheming villain or a relevant joke. For the evil candidate, the clock answer invented a named blacksmith, and the troll answer changed the bargain without adding villainous character. Funny produced essentially the same changes. On selection scenes, the top candidate changed a rival-party answer into generic defense advice. The same neuron winning both searches at similar doses is evidence that this ranking is not isolating the two intended behaviors. The prompt-only villain comparison did change voice but also invented facts and sometimes hit the token cap. Complete generations and matched random controls are in [results/evil_revised.json](evil_revised.json) and [results/funny_revised.json](funny_revised.json).

No focused Llama candidate was promoted to the runtime registry. That Llama search ended after this bounded validation because its generated answers did not support the trait claim.

## Qwen 3.5 4B: villainy, comedy, and distributed probes

The later Qwen experiment used the cached `mlx-community/Qwen3.5-4B-4bit`
checkpoint. Its local text model has 32 layers, 9,216 post-SwiGLU MLP
coordinates per layer, and a 2,560-dimensional residual stream. Qwen mixes
linear and full attention layers. The same eight authored training scenes,
three selection scenes, and five held-out validation scenes were used for each
target. Generations used temperature 0.7, a fixed seed per call, and a 64-token
cap. All interventions were transient; weights were unchanged. The seven
original tests plus an MLP-output-shift test pass, and a zero-dose Qwen hook
reproduced baseline text exactly. Baselines remained identical across all Qwen
experiment files after many intervention swaps.

The judge was `deepseek-ai/DeepSeek-V4.1-Flash` through Together AI. It saw
scene facts, the player prompt, and the reply, but no experimental condition.
A **persona hit** means target strength at least 2 on its 0–3 rubric; a
**grounded hit** additionally needs relevance and grounding at least 1 on
their 0–2 rubrics. The same prompt-only reply received different relevance or
grounding grades in separate batches, so the counts are screening judgments,
not a stable human evaluation. Every text and reason is saved in the linked
JSON files.

| Held-out intervention | Villain persona hits | Comedy persona hits | Matched controls |
| --- | ---: | ---: | --- |
| Published L14 N8949 positive / L13 N7956 negative, per-input dose 1 | 2/5 | — | Four random pairs: 0/5 each |
| Same pair, selection-chosen dose 0.25 | 0/5 | — | Four random pairs: 0/5 each |
| Selected single neuron | 0/5 | 0/5 | Four random neurons: 0/5 each |
| Selected sparse set, 8 or 32 MLP coordinates | 0/5 | 0/5 | Two matched random sets: 0/5 each |
| Selected MLP-output direction | 0/5 | 0/5 | Villain random directions 0/5; one comic random direction 1/5 |
| Selected attention-output direction | 0/5 | 0/5 | Villain random directions 0/5; one comic random direction 1/5 |
| Explicit persona prompt | 5/5 | 5/5 | No neural intervention |

The reported Qwen pair is the interesting result here. Its full-dose
intervention **clearly changes the character of some replies toward murderous
villainy**, which is a real qualitative persona effect. One of those two
villain-persona hits met the stricter relevance and grounding thresholds.
Other replies became incoherent or drifted away from the scene. The first
fixed-calibration pilot found one persona hit; per-input calibration is the
closer implementation of the [reported recipe](https://x.com/_lyraaaa_/status/2113833330429083662).
Calibration values were around 21 activation units per signed unit dose for
both coordinates and varied slightly by prompt. See
[per-input full dose](qwen_known_recipe_dynamic.json) and
[fixed-calibration pilot](qwen_known_recipe.json).

The pair ablation tested each neuron alone and the pair at doses 0.25, 0.5,
and 1.0 on selection scenes. Only the pair at 0.25 had a judged persona hit
(1/3); neither member alone did. The quarter-dose pair then had 0/5 hits on
held-out scenes. This does not establish a stable nonlinear interaction.
The [ablation](qwen_known_pair_ablation.json) and
[quarter-dose validation](qwen_known_pair_quarter.json) contain the
full texts.

The single-neuron screens used contrastive teacher-forced MLP activations and
authored-response log odds. Their selected coordinates were L30 N556 for
villainy and L31 N437 for comedy; neither generalized to a persona hit.
The distributed screens ranked eight- or 32-neuron sets, MLP-output
directions, and attention-output directions from contrastive activations.
Each family's best selection candidate was separately tested on all five
held-out scenes with two matched random controls. None beat its controls on
persona hits. Results: [villain single](qwen_villainy_single.json),
[comedy single](qwen_comedy_single.json),
[villain direction screen](qwen_villainy_directions.json),
[comedy direction screen](qwen_comedy_directions.json),
[all villain methods](qwen_villainy_all_methods.json), and
[all comedy methods](qwen_comedy_all_methods.json).

These tests do not exhaust distributed mechanisms. The authored contrasts
share wording and topic cues with the target behavior, the selection set is
small, and one seed cannot establish reliability. A useful next localization
step would patch residual or component activations from a villain response
into the same neutral prompt token by token, then test whether the resulting
behavior survives new topics and seeds. An existing
[Qwen3.5-4B layer-18 TopK SAE](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk)
could provide feature directions; it was not downloaded or tested in this
bounded run. Its model card describes a single-layer residual SAE, so any
candidate feature would still need causal intervention and held-out validation.
