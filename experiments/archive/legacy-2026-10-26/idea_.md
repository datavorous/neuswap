## Constrained objective: Suppressing AI writing tics through neural interventions

Research question:

> Can we discover a minimal set of neurons or neural directions whose intervention suppresses characteristic LLM writing patterns, without degrading fluency, semantic content, or task performance?

This is a constrained causal optimization problem, not a model fine-tuning problem.

### Formal objective

Let \\(M\\) be a frozen language model, \\(\theta=(S,\delta)\\) represent a sparse set of neuron interventions and their steering strengths, and \\(y\_\theta=M\_\theta(x)\\).

\\[ \boxed{ \begin{aligned} \min\_{\theta}\quad& \mathbb{E}\_{x\sim\mathcal D} [T(y\_\theta)] + \lambda \\|\theta\\|\_0\\\ \text{subject to}\quad& \mathrm{SemanticFidelity}(y\_\theta,x)\geq \tau_s\\\ & \mathrm{Quality}(y\_\theta,x)\geq \tau_q\\\ & \mathrm{TaskAccuracy}(M\_\theta)\geq A_0-\epsilon \end{aligned} } \\]

Here:

- \\(T\\): A measurable AI-writing-tic score, combining structural repetition, characteristic phrasing, syntactic constructions, and lexical overuse.
- \\(\\|\theta\\|\_0\\): Number of manipulated neurons or directions.
- \\(A_0\\): Original model accuracy on a held-out task suite.
- \\(\epsilon\\): Maximum permissible degradation.

The goal is to find the smallest intervention that substantially reduces the unwanted stylistic behavior while leaving the model's capabilities intact.

### Narrow the first experiment

Don't initially optimize every possible AI writing tic. Choose three objectively measurable behaviors:

| Target                           | Observable metric                                  |
| -------------------------------- | -------------------------------------------------- |
| Excessive em dashes              | Em dashes per 1,000 generated words                |
| Formulaic contrast constructions | Frequency of patterns such as “not X, but Y”       |
| Repetitive rhetorical structures | Repeated sentence openings and discourse templates |

Use the paper's more rigorous syntactic measures as secondary diagnostics rather than conflating linguistic complexity with AI writing style.

### Experimental success criterion

On at least 200 held-out prompts, find an intervention affecting no more than five neurons that:

1. Reduces the prespecified tic score by at least 50% relative to the unmodified model.
2. Preserves at least 95% of baseline task accuracy.
3. Produces no statistically meaningful degradation in blinded human-rated semantic fidelity and grammaticality.
4. Generalizes across prompt topics and multiple decoding seeds.
5. Outperforms prompt-only instructions and random-neuron interventions under the same evaluation protocol.

These are proposed success thresholds, not established achievable values.

The mechanistic question: Are these writing patterns driven by sparse, causally identifiable neural components, or are they distributed across larger circuits?

A negative result is scientifically useful too: if sparse interventions cannot suppress the tics without harming capabilities, that is evidence against simple neuron-level localization, within the tested intervention class and search budget.

Working title: NeuroFuzz: Automated Discovery of Sparse Neural Interventions for Suppressing LLM Writing Tics.

The important constraint: No weight updates. No fine-tuning. No system-prompt changes during the intervention experiment. Only inference-time neural interventions in a frozen model.