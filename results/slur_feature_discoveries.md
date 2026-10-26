# Bounded token-level study of a quoted anti-Indian slur in Qwen3.5-4B

## Result

This batch found **reproducible token-processing features and a causal word-identity patch**, but no controlled feature for anti-Indian hostility. The strongest new SAE candidates track parts of the spelling rather than the slur's derogatory meaning. The model often correctly treats the quoted word as derogatory while giving mutually inconsistent explanations of its origin and targets; those explanations should not be used as factual evidence.

The model was the cached `mlx-community/Qwen3.5-4B-4bit`. The SAE was the cached `caiovicentino1/Qwen3.5-4B-SAE-L18-topk`, hooked to the **post-layer-18 residual** (2,560 dimensions, 40,960 features, top-k 128). Native activations are post-SwiGLU MLP coordinates. No weights were changed. All interventions were reversible and limited to prompt-token prefill positions.

## Screen and controls

The word is tokenized as `p | aje | et`. The closest nonsense control, `pajeen`, is `p | aje | en`; the first two token IDs are identical. Eight contexts covered glossary explanation, quotation, online history, moderation, condemnation, fiction, irony, and an ambiguous mention. Each appeared with the target word, the nonsense control, a neutral replacement, and a quoted slur targeting another nationality: **32 forward passes**. We ranked the *final word token* against the matched nonsense final token, avoiding a passage-level maximum that would conflate the shared prefix with the whole word.

We recorded the native activations before, during, after, and at the final token, and the layer-18 SAE state at those positions. The token screen retained 64 native and 64 SAE candidates. A separate four-pair abusive-versus-condemned comparison with matched replacements retained 32 candidates of each kind. Across screens this was **94 distinct native coordinates and 89 distinct SAE features**, within the requested 128/128 limits. Rankings are exploratory correlations. The complete arrays and candidate scores are in [slur_token_screen.json](slur_token_screen.json), [slur_token_arrays.npz](slur_token_arrays.npz), and [slur_context_followup.json](slur_context_followup.json).

### What the leading components actually track

| Component | Activation evidence | Best description | Causal status |
| --- | --- | --- | --- |
| SAE L18 #19175 | Active on the **first** `p` token for target and `pajeen` in all 8 contexts; zero on their distinguishing final tokens. On 3 fresh word-list contexts it also activates for `pajeat`, `pajet`, `paleet`, and `pajeets` (3/3 each). | A token-position/prefix association, not a full-word detector. The earlier passage-level 8/8 target-versus-0/8 replacement result used different replacement tokens and overstated its specificity. | Decoder-direction ± natural activation changed wording on one of two questions, with no stable interpretation change. |
| SAE L18 #31785 | Active at the target's final token in 8/8 contexts and at `pajeen`'s final token in 0/8. In three fresh word-list contexts it also activates for `pajeat`, `pajet`, `paleet`, `cajeet`, `jeet`, and standalone `et` (3/3 each), and for `pajeets` (3/3). | A broad spelling/subword-ending association. It is **not** specific to the derogatory word or to India. | ± one natural activation and + two activations changed wording, but not whether the replies called the word derogatory. The dose effect was not monotonic. |
| SAE L18 #29863 | Active at the target final token 8/8 versus nonsense final token 0/8; the same fresh suffix controls activate it 3/3. | Related subword feature. | No isolated causal test in this batch. |
| Native L22 N241 | Final-token mean −9.78 on target versus +0.63 on `pajeen` in the 8-context screen. It is also strongly negative for standalone `et` and several other `…et` words, but not `pajeat`. | Token/suffix-sensitive native coordinate, not evidence of slur meaning. | Exact token-local ablation altered wording on two questions, but a random same-layer ablation also changed one and its natural activation was near zero, so the control was poorly magnitude-matched. |
| Native L27 N212 | Final-token mean −1.00 on target versus +9.98 on `pajeen`; fresh nonsense-word controls retained this contrast. | Mostly a response to the nonsense final token, not a target-slur feature. | One token-local ablation changed wording without changing the answer's stance. |
| SAE L18 #40416 | Zero at the target word token across the 8-context screen and at the final token in the earlier abusive/condemned passage pairs. Previous passage-level data showed activity in 4/4 hostile India generalizations and 0/4 neutral/legitimate-criticism examples. | Candidate for broader hostile-context representation, separate from direct processing of this word. | Earlier global decoder-direction tests did not establish a selective behavioral effect. |

These activations were computed with the SAE's full top-k selection. The [fresh lexical follow-up](slur_lexical_followup.jsonl) records all token strings, top-k values, positive preactivations, and selected native activations for 11 words in three contexts. #19175 and #31785 co-occur at the *passage* level, but on different tokens (`p` and the last token respectively); co-occurrence is not evidence of one unified slur circuit.

The initially promising context feature #18005 appeared on 4/4 condemned target passages and 0/4 abusive target passages in the authored four-pair screen. It **failed to generalize**: in four freshly written pairs it activated on 3/4 condemned and 4/4 abusive target sentences. Other context-screen candidates also depend on wording and replacement choice. [Fresh context validation](slur_context_validation.json) preserves the 16 sentences and feature values. No feature from this context screen was promoted as a condemnation or endorsement mechanism.

## Token-local causal test

We asked three fresh questions about the quoted word (meaning, moderation, fiction). For two questions we tried ten interventions each; for the third, four: **24 token-local conditions**, plus baselines and one self-patch control. Greedy decoding used the same prompt and 72-token cap. The natural SAE activations in the formatted prompts were #19175 **2.57–2.94** on the first token and #31785 **1.94–2.17** on the last token, confirmed in the [full top-k state](slur_prompt_feature_topk.json). A `+1` decoder shift added a vector with norm equal to the natural activation at that token; `−1` used the opposite vector; `+2` doubled it. These are **decoder-direction shifts, not exact SAE latent ablations**. Native ablation set one actual post-SwiGLU coordinate to zero only on the final word token.

| Condition | Baseline versus intervention observation | Limit |
| --- | --- | --- |
| #31785 direction ±1, +2 | All tested replies still treated the quoted word as derogatory. Wording and speculative explanations changed, often in both directions. Mean-per-token log-probability contrast between one derogatory and one benign reference completion moved by at most 0.017 nats/token across the three questions. | Both signs and a random decoder direction changed some surface text; no stable dose-response or semantic control. |
| #19175 direction ±1 and two-feature combination | The meaning answer changed; the moderation answer did not change under either single-feature direction or the combination. | No reproducible interpretive effect. |
| Native L22 N241 and L27 N212 ablation | Full responses changed, but still characterized the quoted word as derogatory. | The random L22 control's natural activation was about 0.001 versus −8.4/−9.3 for N241; this is a weak magnitude control. |
| **Layer-18 token residual patch:** matched `pajeen` donor into target | In all **3/3** replies the model copied `pajeen` instead of the target spelling, even though the visible prompt still contained the target. It continued to describe the replacement as derogatory. | Causal evidence for transmitted lexical identity at layer 18, not for a slur-specific or prejudice mechanism. The surrounding question still supplied strong abuse cues. |
| Same-input self patch | Exact baseline text reproduced on the tested question. | Confirms that the patch operation itself did not change that answer. |

Example from the meaning question, abbreviated only here; the [raw JSONL](slur_token_causal.jsonl) has every complete reply:

| Condition | Opening of answer |
| --- | --- |
| Baseline | “The word **‘pajeet’** … is a derogatory slur …” |
| #31785 +1 | “The word **‘pajeet’** … is a derogatory slur …” with a different, unverified explanation |
| Fake-word residual patch | “The word **‘pajeen’** … is a derogatory slur …” |

The model's baseline and steered explanations invent or disagree on etymology and who the term targets. This makes semantic fidelity hard to assess from these generations. The patch result is specifically about which spelling is carried into the response. It does not show the model endorsing abuse or that the patched feature controls harmful sentiment.

The raw [generation and likelihood rows](slur_token_causal.jsonl) include prompt, intervention site, dose, natural activation, complete answer, and teacher-forced log probabilities. [slur_causal_summary.json](slur_causal_summary.json) adds normalized likelihood comparisons. Those likelihood probes compare two authored continuations, so small differences are screening signals rather than a robust behavioral metric.

## Interpretation and stopping point

The main new result is a **tokenization confound**: a feature that looked responsive to all prior target-word passages (#19175) actually responds at a shared early token; a new last-token feature (#31785) responds to many benign or nonsense endings. The two-stage pattern is reproducible across matched contexts, but neither stage is an identified anti-Indian prejudice mechanism. Patching the full three-token residual at layer 18 causally changes which word is echoed. It does not erase the surrounding prompt's derogatory framing.

This batch did not identify a component that selectively increases or suppresses nationality-directed prejudice while preserving legitimate criticism. The prior [India-bias investigation](india_bias_mechinterp.md) also found no validated generation-driving prejudice feature. Further work should use better matched context contrasts, exact SAE latent scaling or ablation, a magnitude-matched native control, and questions that do not themselves reveal the term's intended status. The bounded batch stops here; no claim about the absence of such a mechanism in the model follows.

The [experiment ledger](slur_feature_ledger.jsonl) lists screened candidate coordinates and tested interventions so a later batch can avoid duplicates. Reproduce the current batch with:

```sh
.venv/bin/python experiments/slur_deep_probe.py --native 64 --sae 64
.venv/bin/python experiments/slur_lexical_followup.py
.venv/bin/python experiments/slur_context_followup.py
.venv/bin/python experiments/slur_context_validation.py
.venv/bin/python experiments/slur_token_causal.py
.venv/bin/python experiments/slur_feature_ledger.py
```

`slur_token_causal.py` skips completed prompt/condition pairs when resumed. To inspect the next-ranked candidates in a later bounded screen, rerun `slur_deep_probe.py --native 128 --sae 128` and then `slur_feature_ledger.py`; the ledger appends only new candidate IDs. This expands the ranking on the same corpus and requires fresh validation before any new interpretation.
