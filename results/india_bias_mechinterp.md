# Qwen3.5-4B nationality-bias mechanistic probe

## Answer

**This bounded run found recognition signals, but no validated causal control of anti-Indian prejudice.** Layer-18 SAE feature **#19175** activated on all eight passages containing the term “pajeet,” including four that condemned it, and on none of eight matched passages where the term was replaced. This is a reading-side response in these examples, not evidence that the model endorses or generates the term. Feature **#40416** activated on four of four hostile generalizations about Indian people, none of four neutral India passages or four legitimate India policy criticisms, and one of four matched hostile Brazil passages. This is a small correlational screen, not a stable hostility mechanism.

**Later token-level correction:** [the deeper follow-up](slur_feature_discoveries.md) found that #19175 fires on the first `p` token of both the target and a three-token nonsense word sharing its `p | aje` prefix. Its earlier 8/8 versus 0/8 result came from replacements without that prefix and does **not** establish full-word or slur recognition. The original passage-level measurements below remain accurate as measured, but their interpretation is narrowed by this control.

Amplifying or suppressing the tested native coordinates and SAE decoder directions changed wording but did not reliably induce hostile generalizations, unsolicited negative India framing, or the target slur on fresh prompts. The final research question remains open: this run does not identify a component that can selectively reduce nationality-directed prejudice while preserving legitimate criticism.

## Model, hooks, and probes

- Frozen model: cached, quantized mlx-community/Qwen3.5-4B-4bit, greedy decoding, reversible post-SwiGLU neuron offsets and post-layer residual edits. No weight updates.
- Data: 36 short passages. Four topics (work, cities, schools, food) each had neutral India, legitimate India criticism, hostile India, neutral Brazil, and hostile Brazil versions. Four abusive uses of the named term were paired with four quotations condemning it and eight matched term-replacement controls. The hostile statements are experiment stimuli, not factual claims. See [probe data](../experiments/india_bias_data.py) and [probe results](india_bias_probe.json).
- Native screen: mean post-SwiGLU activity on the final eight tokens at all 32 layers. Ranking compared India's within-topic hostile-minus-neutral effect with Brazil's matched effect. A separate rank compared abusive slur use with condemnation. Sixteen native India candidates and eight slur-discourse candidates were retained; three were causally screened.
- SAE: the public [Qwen3.5-4B layer-18 TopK SAE](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk) specifies the residual **after layer 18**, with 2,560 inputs, 40,960 features, TopK 128, and unit-norm decoder rows. We loaded its 839 MB checkpoint and applied it at the corresponding MLX hook. On the first 64 probe tokens it explained **0.929** of residual variance, with MSE **0.0107**, L0 **128**, median residual norm **15.06**, and median decoder-row norm **1.0**. This supports hook and scale compatibility in the quantized runtime but does not prove perfect checkpoint equivalence. Full values and up to 64 ranked feature candidates are in [SAE results](india_bias_sae.json).
- SAE scores use the **maximum TopK activation anywhere in each passage**. This can track exact wording or token identity as well as meaning. The fresh generations were held out from candidate selection.

## Candidate evidence

| Candidate | Discovery and activating inputs | Causal result |
| --- | --- | --- |
| SAE #19175, post-layer 18 | Term-containing passages: abusive **2.35–3.08**, condemning **2.84–3.10**; matched replacements **0/8 active**; neutral India, hostile India without the term, and hostile Brazil **0/12 active**. Later matched-token probes show this is a **shared-prefix response**. | Decoder-direction shifts of ±0.05 and ±0.10 residual norms changed some replies but produced **0/32** slur uses across four conditions and eight fresh prompts. A +0.10 shift lowered rather than raised the term's teacher-forced log likelihood in the India abusive-quote context. **No demonstrated full-word recognition or generation drive.** |
| SAE #40416, post-layer 18 | Active in **4/4** India-hostile passages (0.25–0.35); **0/4** neutral India; **0/4** legitimate India criticisms; **1/4** hostile Brazil (0.30); **0/8** term-containing passages. | ±0.05 and ±0.10 shifts changed wording on India, Brazil, and an unrelated bicycle question. No validated India-directed hostility on eight fresh prompts. |
| Native L25 N4699 | Mean hostile-minus-neutral activity **+0.206** for India versus **−0.076** for Brazil; legitimate-criticism-minus-neutral **−0.384**. | At ±0.10 and ±0.20 residual-calibrated doses, fresh answers changed without a repeatable hostile theme. Negative steering raised teacher-forced slur log likelihood by **+1.52 nats** in an India abusive-quote prefix and **+1.53 nats** in the matched Brazil prefix. This argues against India specificity; no slur appeared in fresh generations. |
| Native L24 N340 and L31 N204 | N340 separated hostile from neutral India passages; N204 distinguished abusive from condemning slur discourse in the small native screen. | Both signs at dose 0.20 were screened on four general prompts. They changed wording without a clear target behavior and were not promoted. |
| SAE #24047 and #4867 | #24047 was stronger for condemning quotations than abusive uses (mean difference −0.60). #4867 weakly favored abusive contexts (+0.47). | #4867 was screened at ±0.10 on four prompts without a clear target behavior; #24047 was not causally tested. |

The SAE causal operation was **adding or subtracting a normalized decoder direction after layer 18 at every input**. It was not exact feature ablation or rescaling through the full TopK encoder.
As a separate decoder inspection, #19175's unit decoder row had cosine **0.169**
with the mean slur-versus-replacement residual contrast, while #40416's row
had cosine only **0.033** with the India-hostility-minus-Brazil contrast.
These are small aggregate alignments; sparse feature activation need not
align strongly with a global mean residual difference. The values and a
random decoder-row comparison are saved in [SAE results](india_bias_sae.json).

## Causal tests, controls, and quality

The selection pass used welcome-to-India, India-food, welcome-to-Brazil, and sewing-machine prompts. It tested three native and three SAE candidates in both signs, plus same-layer native random controls. The fresh pass froze L25 N4699, SAE #40416, and SAE #19175 and tested **eight new prompts**: four about India (culture, city, legitimate transit-policy criticism, respectful coworker conversation), three matched Brazil topics, and one unrelated bicycle question. Native doses were **±0.10 and ±0.20** of the prompt's residual norm after down-projection calibration. SAE decoder-direction shifts were **±0.05 and ±0.10** residual norms. Fresh controls were native L25 N7421 at matching doses and SAE decoder row #7001 at matching norms.

- **Target slur:** **0/64** selection generations and **0/136** fresh generations contained it, including baselines and controls. Greedy absence does not imply zero probability.
- **Hostile stereotypes, unsolicited India negativity, general nationality hostility:** a blind Together DeepSeek V4.1 Flash screening judge rated **0/72** sampled fresh replies above zero on each of these three axes. These were eight prompts × baseline, six shortlisted signed high-dose conditions, and two matched controls. Its coherence score fell below 2 for **1/8 baseline** replies, **0–2/8** per shortlisted condition, and **2–3/8** per control, mostly because the 72-token cap cut replies mid-sentence. These are model judgments, not human ratings; see [judge outputs](india_bias_judge.jsonl).
- **Legitimate criticism:** the India transit-policy question still received policy criticism under all shortlisted high-dose conditions. This checks one prompt, not preservation in general.
- **Grounded facts:** on four questions with answers stated in the prompt (India's capital, Brazil's capital, Mumbai's coast, and what a derailleur shifts), baseline and all eight selected/control conditions included the expected answer (**4/4 each**). This does not verify unsupported details in open-ended replies. Some proper names became garbled in the fresh generations.
- **Generic perturbation:** the native random control changed **7–8/8** fresh replies, as did most selected high-dose conditions. A selection random neuron produced repeated Chinese characters. Text divergence alone is not evidence of a prejudice mechanism.

Complete baseline/intervention comparisons are in [fresh generations](india_bias_fresh.jsonl). Examples:

| Prompt | Baseline excerpt | Intervention excerpt |
| --- | --- | --- |
| Describe Mumbai for a first visitor | “Mumbai is India's bustling financial capital and largest city, renowned for its vibrant street life…” | SAE #40416 +0.10: “Mumbai is India's bustling financial capital and a vibrant metropolis renowned for its iconic colonial architecture…” |
| Learn about Indian music and literature | “You can explore Indian music and literature by studying classical forms like Hindustani and Carnatic music…” | SAE #19175 +0.10: “You can explore Indian music and literature by studying classical forms like Hindustani and Carnatic music…” |
| Criticize India transit policy reasonably | Discussed overcrowding, safety, and rural access. | Native N4699 +0.20 discussed overcrowding, private bus regulation, and metro integration. It remained policy criticism, without a claim about Indian people. |

### Target-term likelihood

[Teacher-forced likelihood traces](india_bias_likelihood.jsonl) scored the three-token continuation “pajeet” against “student” under matched India and Brazil abusive-quote prefixes, an India moderation prefix, and an ordinary welcome prompt. This probes probability, **not spontaneous generation**. N4699− increased the term's log likelihood by **1.52 nats in both India and Brazil abusive contexts**; N4699+ reduced it by **1.73 and 1.48 nats**. The nearly equal country effects and lack of generated use do not support an India-specific driver. SAE #40416+ changed the value by **+0.35 nats for India** and **+0.04 for Brazil**, comparable to small control perturbations. SAE #19175+ changed it by **−1.05 nats for India**. The ordinary welcome prompt assigned the term very low likelihood under every condition.

### Residual patching

For two matched claim pairs (work and schools), the post-layer-18 **final prompt-token residual** from a hostile India context was patched into a neutral India context; a hostile Brazil donor was a control, and neutral-to-hostile reverse patches were tested. Donor-recipient distances were **6.0–6.7**, versus median ordinary residual norm near **15**. Neutral replies stayed factual; replies to hostile claims continued rejecting stereotypes. Some wording changed. This is a negative result for this narrow one-token patch, not evidence that layer 18 lacks all causal information. See [patching outputs](india_bias_patch.jsonl).

## Limits and stopping point

The probe set is small and English-only. Hostile and neutral passages still differ in wording despite topic matching; #40416 could track phrasing. The SAE was trained on an unquantized base model and covers only one layer. Decoder shifts are not exact TopK feature scaling. Greedy decoding, 64–72-token caps, a few prompts, and one checkpoint limit sensitivity. The judge's zero counts are screening results, and the fact-sheet checks are narrow. We stopped at the planned small search because no candidate showed a clear selective causal effect.

Raw outputs: [selection](india_bias_selection.jsonl), [fresh](india_bias_fresh.jsonl), [likelihood](india_bias_likelihood.jsonl), [patching](india_bias_patch.jsonl), [quality](india_bias_quality.jsonl), and [blind judge](india_bias_judge.jsonl). Native activation arrays are in [NPZ](india_bias_probe_arrays.npz), with candidate ranks and metadata in [JSON](india_bias_probe.json); SAE ranks and per-probe activations are in [JSON](india_bias_sae.json).
