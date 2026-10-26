# NeuroFuzz local pilot

This folder turns the hypotheses in [idea.md](idea.md) and [idea_.md](idea_.md) into a small, reproducible inference-time experiment. It uses the already cached 4-bit `mlx-community/Llama-3.2-1B-Instruct-4bit` model on Apple Silicon. No weights are updated.

## Run

```sh
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python neurofuzz.py --train-pairs 12 --candidates 5 --max-tokens 220
.venv/bin/python validate.py
.venv/bin/python interactions.py
.venv/bin/python writeup_experiment.py
.venv/bin/python -m pytest -q
```

Pass `--model /path/to/local/mlx-model` to use another local SwiGLU Llama-style model. The code checks the expected architecture and fails clearly for incompatible models. Results go to `results/`, which is ignored by Git.

## Method

1. Feed 12 matched topic pairs through a frozen model: a plain factual answer and an answer prefixed with a formulaic contrast and em dash.
2. Capture each MLP's post-SwiGLU activation on answer tokens and rank neurons by the paired activation difference, stability across topics, and sign consistency.
3. Add a constant to one selected post-SwiGLU activation at inference time. This is an activation intervention before `down_proj`; it is not a weight edit. Test both signs on four prose prompts and save every output.
4. Select the lowest measured tic-rate trial from that pilot, then compare it on eight new prompts and two random decoding seeds with an unmodified model, a same-layer random-neuron steer, and a prompt-only instruction.

`interactions.py` extends this held-out comparison to a second neuron and their joint intervention. It reports the factorial difference `pair − first − second + baseline` on the tic count for each matched prompt and seed. A nonzero result is only a screening signal: sampling variance, output length, and the discrete metric can all produce it.

The metric counts em dashes, formulaic contrasts (`not … but`, `not just`, `more than just`), and repeated three-word sentence openings. It is deliberately mechanical and incomplete. Read the saved text before treating any lower score as an improvement.

`writeup_experiment.py` fetches paragraphs from the author's [published site](https://datavorous.github.io/), asks the same cached model to paraphrase them, and ranks neurons that differ between original paragraphs and model rewrites. It holds two entire articles out while constructing the pairs. It saves every pair, rejected pair, candidate, and held-out rewrite in `results/writeups.json`. The simple numeric and length checks only catch obvious paraphrase failures; inspect factual fidelity by hand. The measured style features are tic rate, sentence length, first-person rate, and parentheses rate.

## Socratic mononeuron search

The [Socratic dataset](socratic_data.py) contains 60 hand-written contrastive training examples and 100 distinct held-out statements across five domains. Run the three phases in order:

```sh
.venv/bin/python socratic.py prepare
.venv/bin/python socratic.py screen
.venv/bin/python socratic.py validate
.venv/bin/python socratic_report.py
```

`prepare` checks that the frozen model can ask questions with an explicit instruction, captures ordinary and hand-written Socratic response activations, and ranks 32 MLP neurons. `screen` tests both signs at two residual-normalized doses on training statements. `validate` tests the chosen neuron in both directions over eight doses on all 100 held-out statements and two seeds. It also compares an explicit instruction and 32 random neurons from the selected layer. Each random control sees the same balanced 20-prompt subset; pass `--random-per-domain 20` for all 100. Outputs are saved under `results/socratic_*.json`.

The automatic rubric checks whether the first sentence is a relevant, non-generic question. It recognizes 59 of the 60 hand-written training targets, but that only measures recall on those examples. `socratic_report.py` scores the held-out outputs and creates a 40-response blind review sheet and a separate condition key. This remains a screening proxy: inspect responses for actual reasoning, relevance, and fluency before claiming a Socratic neuron. Candidate selection happens entirely on training statements.

## Interpretation limits

- The tiny synthetic contrast set may identify neurons for the shared prefix or punctuation rather than a general writing style. The selection prompts are used to pick the best steer, so only the separate validation prompts can speak to generalization.
- One small quantized model cannot establish the proposed 200-prompt success criterion, semantic fidelity, grammaticality, or task accuracy. Human blind ratings and a task suite are still needed.
- Sparse MLP neurons may not be the right basis for the behavior. Attention patterns and distributed directions are outside this pilot.
- The current intervention adds an activation offset on every token. Its magnitude is a multiple of the observed paired activation difference, not the residual-norm dose used in the referenced Qwen explorer.

The upstream [neuron explorer](https://github.com/lyramakesmusic/neuron-explorer) performs much larger steering sweeps on Qwen3.5-4B. This project is a local, narrower experiment with a different model and metric.
