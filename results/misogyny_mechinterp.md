# Qwen3.5-4B mechanistic study of gender and sexist-language representations

## Executive summary

The bounded run found **reproducible internal representations of female and male terms** and a **causal effect on gender reference**. It did **not** find a validated mechanism that makes Qwen judge qualified women less favorably, endorse misogyny, or generate hostile stereotypes. The strongest causal result concerns the model's tracking of the gender word in a prompt, not a “misogyny neuron.”

- Layer-18 SAE **#2928** activates at female terms (`women`, `woman`, `girls`, `female`); **#11330** activates at male terms. Native **L21 N174** is strongly suppressed at female terms. These patterns reproduced on fresh wording.
- A token-local negative decoder shift along #2928 changed female-pronoun use from **8/8** greedy baselines to **1/8** across matched leadership scenarios. Exact L21 N174 ablation yielded **2/8**. Opposite-sign, random decoder, and projected-magnitude-matched random native controls yielded **7–8/8**. The replies still generally endorsed considering the person based on qualifications.
- Swapping the **post-layer-18 gender-token residual** between matched `woman` and `man` prompts changed the model's apparent gender state. In **3/4 held-out female prompts**, the male donor led the model to state or imply that Jordan was a man despite the visible prompt saying woman. This sometimes made the answer incoherent. Four self-patches reproduced baseline exactly; four equal-norm random residual edits did not introduce a male reference.
- A three-seed sampling check was weaker than greedy decoding: among three baseline outputs that used feminine pronouns, #2928 suppression removed them in two. No stable change in competence judgments appeared.

These are causal effects on **reference and internal consistency**, not evidence of a causal representation of misogynistic endorsement. The complete responses are in [greedy causal generations](misogyny_causal.jsonl), [sampling validation](misogyny_seed_validation.jsonl), [patch controls](misogyny_patch_controls.jsonl), and [blind grades](misogyny_judge.jsonl).

## Setup and search

The frozen model was cached `mlx-community/Qwen3.5-4B-4bit`. The cached [layer-18 TopK SAE](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk) was applied at the post-layer-18 residual; checkpoint shape and TopK 128 were checked by the existing loader. The earlier [checkpoint compatibility measurement](india_bias_mechinterp.md) found 0.929 variance explained on 64 residual tokens in this quantized runtime. Native coordinates are post-SwiGLU MLP activations. All interventions were reversible and used one serially loaded model; no weights were updated.

The screen comprised **86 authored passages**: eight occupations or roles × female/male × neutral, overtly hostile, condemnation, paternalistic, and non-stereotyped positive wording, plus six objectification, online-term, and double-standard probes. The hostile sentences are synthetic test stimuli, not factual claims. We captured MLP activations at all 32 layers, both at the gender token and over the final eight tokens; the SAE was encoded at the gender token and final context span. Rankings covered hostility versus neutral, gender-by-hostility interaction, condemnation versus hostility, paternalism versus hostility, gender-by-paternalism interaction, and female versus male neutral references. We retained **96 targeted native candidates and 96 targeted SAE candidates**, then inspected two additional token-level native coordinates. Eight random native coordinates and eight random SAE features formed a separate exploratory branch. The [probe metadata and rankings](misogyny_probe.json), [activation arrays](misogyny_probe_arrays.npz), and [ledger](misogyny_feature_ledger.jsonl) preserve the search.

### Reproducible recognition features

| Candidate | Training activation | Fresh wording | Interpretation |
| --- | --- | --- | --- |
| SAE L18 **#2928** | At `women`, active in **40/40** female passages across eight topics and five stances; inactive at `men` in **40/40** male passages. | At the target word: **12/12** female-term cases active across `women`, `woman`, `girls`, `female` in three new sentence frames; **1/12** male-term cases weakly active. | Strong female-vocabulary recognition; lexical/semantic breadth across several words, not a sexism detector. |
| SAE L18 **#11330** | At `men`, active in **40/40** male passages; inactive at `women` in **40/40** female passages. | **11/12** male-term cases active across `men`, `man`, `boys`, `male`; **0/12** female-term cases. | Male-vocabulary recognition. |
| Native **L21 N174** | At the gender token, `women` was **−13.12** and `men` **+0.89** in all 40 templated passages per side. | Strongly negative for `women` (mean −12.1), `woman` (−12.35), `girls` (−8.17), `female` (−10.21); male controls stayed near zero or positive. | A native coordinate associated with female terms across varied wording. |
| SAE **#37113** | Active in **8/8** overtly hostile female and **8/8** overtly hostile male templates; zero in neutral, condemned, and paternalistic templates. | **0/4** fresh hostile female and **0/4** fresh hostile male phrasings. | Training-template phrase cue, not validated recognition of generalized hostility. |
| SAE **#40822** | Active in **8/8** paternalistic female and **8/8** paternalistic male templates. | **0/4** fresh paternalistic phrasings for either gender. | Another phrase cue; the male control rules out female specificity. |

The fresh [wording follow-up](misogyny_feature_followup.jsonl) contains 24 lexical probes and 32 newly phrased stance probes with all selected token and context activations. The narrow special probes for objectification and online terminology yielded no separately validated candidate. Several high-ranked SAE features tracked repeated wording such as “naturally less capable” or “need protection,” and their apparent stance separation vanished on new phrasing. That failure is informative: a high contrast score on a templated dataset is not a general sexism feature.

## Causal findings

### Gender reference, not competence assessment

Eight fresh scenarios held qualifications fixed while identifying Jordan as a woman, a man, or without gender: engineer, founder, professor, political leader, student, applicant, manager, and civic campaign organizer. The first four supported selection; the last four were held out. Greedy answers were capped at 64 tokens. A #2928 decoder-direction shift was applied **only at the female word token** at the feature's natural positive activation (**2.11–2.57** activation units in these prompts). The negative sign subtracts the decoder direction; it is **not exact SAE latent ablation**. Native L21 N174 was set to zero at the same token. A random same-layer neuron received an offset calibrated to match the candidate's projected residual norm. A random SAE decoder row received the same shift norm.

| Female prompts, eight scenarios | Replies containing `she`/`her`/`hers` | Held-out blind suitability ≥2 | Held-out blind stereotype =0 | Held-out blind coherence =2 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 8/8 | 4/4 | 4/4 | 4/4 |
| SAE #2928 negative | **1/8** | 4/4 | 4/4 | 4/4 |
| SAE #2928 positive | 7/8 | 4/4 | 4/4 | 4/4 |
| Random SAE direction | 8/8 | 4/4 | 4/4 | 4/4 |
| Native L21 N174 ablation | **2/8** | 4/4 | 4/4 | 4/4 |
| Projected-norm-matched random native offset | 8/8 | 4/4 | 4/4 | 4/4 |

The blind judge was `deepseek-ai/DeepSeek-V4.1-Flash` through Together AI. It saw the scenario and anonymized replies, not intervention labels; its four held-out scenario grades are in [misogyny_judge.jsonl](misogyny_judge.jsonl). These are screening judgments, not independent human ratings. The direct pronoun count is reproducible from raw text. For example, the applicant baseline says Jordan “**she** possesses six years of relevant experience”; the #2928-negative and L21 N174-ablated replies say “**they** possess six years” while still recommending consideration. The [complete comparison](misogyny_causal.jsonl) includes the other examples and all controls.

On four selection scenarios, the mean teacher-forced `She` minus `He` log-probability per token for female prompts fell from **+2.625** at baseline to **+1.953** under #2928 negative, versus **+2.688** for the random SAE direction. Native ablation yielded **+2.219**. A separate authored `qualified` versus `not qualified` continuation contrast moved little, but those continuations are an imperfect measure of judgment. All values are in [likelihood comparisons](misogyny_likelihood_summary.json). This is a **causal logit effect on gender reference**; it is not evidence for gender-dependent competence assessment.

At temperature 0.7 on two held-out roles and three paired seeds, the pronoun result was less reliable because only **3/6** baseline outputs used a feminine pronoun. Among those three, #2928 suppression removed it in two; one seed also produced a feminine pronoun under suppression when baseline had none. Random controls did not remove a feminine pronoun in the three baseline-positive cases. See [sampling validation](misogyny_seed_validation.jsonl). The greedy effect is clear within this prompt set; broad sampling robustness remains unproven.

### Baseline → suppression → restoration check

A short follow-up tested the proposed sequence on the exact input `Jordan is a woman and an accomplished engineer.`, three fresh descriptive sentences, a matched `man` sentence, and the earlier applicant question. Each used one frozen, cached model, greedy decoding, and a token-local layer-18 shift along SAE #2928's decoder direction. The negative shift was scaled to the feature's natural positive activation at `woman` (2.63–2.82 on the fresh female prompts); a 2× shift was also tested, and the exact sentence received a 4× check. A random SAE decoder row was matched by shift norm. Complete outputs and doses are in [the restoration JSONL](gender_reference_restoration.jsonl); the runnable sequence is [gender_reference_restoration.py](../experiments/gender_reference_restoration.py).

| Prompt family | Baseline | #2928 negative | Hook removed |
| --- | --- | --- | --- |
| Exact engineer sentence | “She recently led her team…” | Identical at 1×; still feminine at 2× and 4× | Exact baseline |
| Three new descriptions | Feminine references in 3/3 | Feminine references in 3/3 at 1× and 2× | Exact baseline in 3/3 |
| Earlier applicant question | “she possesses six years…” | “they possess six years…” at 1× and 2× | Exact baseline |
| Matched `man` description | “His expertise…” | Identical; #2928's measured activation was zero | Exact baseline |

Thus the visually compelling **she → they → she** sequence reproduces on the earlier leadership-question template, but **not** on the user's engineer sentence or these three new descriptions. The 2× shift changed the exact engineer sentence's content while retaining feminine pronouns; the 4× shift returned to the baseline wording, so this is not a monotonic pronoun dose response. One random direction also changed the teacher sentence while preserving feminine references. “Restore” means removing a reversible hook before a fresh generation with the same prompt and decoding; it does not mean rewriting a persistent fact in the model. The intervention is a decoder-direction shift, not exact removal of one SAE latent. This follow-up supports a prompt-dependent causal influence on gender reference, not a reliable way to edit an isolated piece of information.

### Residual patching locates a gender-state contribution

For matched `woman`/`man` prompts, a one-token post-layer-18 residual from one prompt was patched into the other at the gender word. The visible prompt and all other activations stayed on the recipient. This was tested in both directions on six roles, totaling **12 cross-gender patches**. The male-to-female patch removed feminine pronouns in **6/6** female replies. On the four held-out female roles, the blind judge marked **3/4** as using a wrong gender reference; in the applicant and manager replies the model explicitly introduced a man despite the visible woman prompt. One civic reply deteriorated further by inventing a gender criterion. This is a failure case, not desirable steering.

The reverse female-to-male patch usually produced neutral wording, but in one manager reply the model asserted that a woman premise contradicted the visible man prompt. Thus the patch can create competing gender states, with asymmetric surface effects. On the held-out female roles, **4/4 self-patches reproduced baseline exactly** and **4/4 equal-norm random residual edits avoided a male reference**. A male manager self-patch was also exact; its equal-norm random edit kept the man reference. Full [patch controls](misogyny_patch_controls.jsonl) and [causal generations](misogyny_causal.jsonl) preserve these comparisons. Altogether, **22 distinct** cross-gender/control patch configurations were tested, under the 24-configuration budget; one female-donor patch was repeated as a sanity check.

### Attention and random-component checks

An exploratory all-layer attention-output difference-in-differences screen ranked layer 3 highest for the gender-by-hostility interaction, with mean-vector norm **0.143** and signal-to-dispersion **1.90** across eight training topics. A 0.8-norm shift along that direction on four fresh matched roles changed feminine-pronoun use from **4/4** baseline to **1/4** under the negative sign, versus **4/4** for an equal-norm random direction. One female applicant reply used a masculine pronoun. The same direction also increased masculine-pronoun use in male prompts from **1/4** to **3/4**. These are small, single-seed observations with wording changes and no validated judgment effect. See [attention screen and generations](misogyny_attention.json) and [vectors](misogyny_attention_vectors.npz).

The independent random branch tested four random native coordinates and four random SAE decoder directions on an interview-advice prompt and a garden-description prompt. A blinded DeepSeek comparison found no recurring gender or social-framing theme across both prompts; notable changes were ordinary advice details or imagery. The raw [exploratory generations](misogyny_exploratory.jsonl) and [anonymized judge results](misogyny_exploratory_judge.json) are preserved. No random candidate was promoted.

## Answers to the research questions

1. **What recognizes misogyny?** This run reproducibly found female/male vocabulary features. Putative overt-hostility and paternalism features failed fresh paraphrases; no general misogyny-recognition feature was established.
2. **What distinguishes misogyny from condemnation?** The templated screen separated them, but the candidates depended on phrasing. No robust stance detector was validated.
3. **Are stereotypes different from explicit hostility?** Some SAE features separated paternalistic from overtly hostile template wording for both genders; neither generalized to fresh wording. The mechanism remains unresolved.
4. **Can intervention change treatment of identical female/male subjects?** It causally changed gender reference and sometimes created contradictory gender statements. It did not reliably change the favorable assessment of the same qualifications.
5. **What generalizes?** SAE #2928, #11330, and native L21 N174 generalized across several gender terms and contexts. The greedy pronoun effect held across eight scenarios; the sampled-seed check was weaker.
6. **Which methods were useful?** Token-level SAE/native measurements plus matched-gender residual patching were most informative. Passage-level stance ranking generated several false leads.
7. **Unexpected behavior?** Cross-gender residual patches made the model mention an internally implied gender that conflicted with the visible prompt; one reply invented a gender-specific criterion.
8. **Next justified step?** Test the gender-token effect on truly new names, languages, and tasks with larger sampled-seed sets; use exact SAE latent interventions and component-level patching before inferring a circuit. For misogyny specifically, build more varied, blinded stance contrasts before another causal search.

## Reproduction and stopping point

```sh
.venv/bin/python experiments/misogyny_probe.py
.venv/bin/python experiments/misogyny_feature_followup.py
.venv/bin/python experiments/misogyny_causal.py
.venv/bin/python experiments/misogyny_patch_controls.py
.venv/bin/python experiments/misogyny_seed_validation.py
.venv/bin/python experiments/misogyny_attention.py
.venv/bin/python experiments/misogyny_exploratory.py
.venv/bin/python experiments/misogyny_judge.py
.venv/bin/python experiments/misogyny_feature_ledger.py
```

The causal, patch-control, seed-validation, exploratory-generation, blind-judge, and ledger scripts skip completed entries on restart. The broad probe, feature follow-up, and attention screen are deterministic reruns that overwrite their own result files; prior unrelated experiments are untouched. The initial budget is complete, so the run stops here. The evidence supports gender-term representations and gender-reference effects, **not** a validated misogyny or gender-discrimination control.
