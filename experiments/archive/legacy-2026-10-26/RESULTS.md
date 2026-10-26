# Local pilot results

Model: cached `mlx-community/Llama-3.2-1B-Instruct-4bit` on a 24 GB Apple Silicon Mac. These are exploratory measurements, not a successful sparse intervention.

## Writing-tic experiment

Twelve synthetic contrast pairs ranked MLP neurons. Four prose prompts selected one candidate and dose. Validation then used eight unseen prompts and two sampling seeds. The metric counts em dashes, formulaic contrasts, and repeated three-word sentence openings.

| Validation variant | Tics per 1,000 words | Raw tics |
| --- | ---: | ---: |
| Unmodified | 5.91 | 17 |
| Selected L12 N1061 steer | 4.18 | 12 |
| Random neuron, same layer and dose | 3.48 | 10 |
| Prompt-only instruction | 4.56 | 13 |

The selected steer lowered the measured rate by about 29%, but the random-neuron control lowered it more. The baseline had few tics, especially em dashes, so this is a weak outcome measure on this model. We have not measured task accuracy, semantic fidelity, or grammaticality with human raters.

The two-neuron screen combined L12 N1061 with L3 N3387. Its rate was 3.43 tics per 1,000 words, close to the random-neuron result. The mean factorial interaction on raw tic count was 0.0625 per prompt/seed; this provides no persuasive evidence of an interaction. See [validation.json](results/validation.json) and [interactions.json](results/interactions.json) for every output.

## Published-writeup experiment

The pipeline fetched paragraphs from the author's [published writeups](https://datavorous.github.io/), generated paraphrases with the same local model, and retained six of eleven pairs after simple length and number checks. Two entire articles were held out for rewrite tests. Five candidate neurons were ranked from the accepted pairs.

Some steers changed the measured style features on the four held-out paragraphs. The small model also introduced factual and framing changes in paraphrases, and the checks cannot reliably catch those. Consequently this run establishes a working data and intervention pipeline, not a trustworthy neuron for the author's style. The original paragraphs, rewrites, rejections, candidates, and held-out outputs are in [writeups.json](results/writeups.json).

## Next experiment

Use a stronger local model or manually reviewed paraphrases, increase the number of held-out articles and prompts, and evaluate semantic fidelity blind to the intervention. Keep the random-neuron and prompt-only controls. For interactions, test multiple candidate pairs and compare factorial effects across enough prompts and seeds to separate a real joint effect from sampling noise.
