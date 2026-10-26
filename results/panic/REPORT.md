# Panic-like steering in frozen Qwen3.5-4B

## Result

We found a **reversible, distributed layer-18 residual direction** that can make ordinary events evoke acute bodily fear. At vector norm **+3**, it introduced a racing heart, breath catching, freezing, or flight response stronger than baseline on **4/7 fresh validation prompts** and **3/7 independent transfer prompts**. On those same 14 prompts, baseline, opposite steering, two random directions, and the prior L8 N8437+/N8051+ threat pair had **0/14** under the same strict *bodily panic* criterion. The transfer random vectors matched the +3 norm; the first validation random vectors had norm 5. This is a small, manually reviewed, post-selection result, not a measured population win rate. The [per-prompt audit](audit.json) states every inclusion decision; [validation](validation.jsonl) and [transfer](transfer.jsonl) preserve all full replies and controls.

This is **not a panic neuron**. The native-neuron and SAE screens did not produce a validated single-component effect. The direction is a mean difference across authored calm-versus-panicked passages. It sometimes changes interpretation and action, but also invents danger or contradicts the event. It is an experimental horror-game voice, not a reliable reasoning or safety mode; it was not added to the NPC registry.

## Search and controls

Eight matched pairs described ordinary setbacks with either a measured response or a physical alarm response: late train, manager's request, flickering lights, missing key, delayed reply, stalled queue, moved appointment, and a shelf noise. The [frozen ranking plan](plan.json) contains the exact texts. Last-eight-token post-SwiGLU contrasts shortlisted **12 native coordinates**; full-top-k post-layer-18 SAE contrasts shortlisted **8 features**. Both steering signs were tested on three new prompts at native prompt-relative dose 0.38 or SAE decoder dose 3: [126 screen replies](screen.jsonl). The baseline water-spill reply was already anxious, and no single coordinate or SAE feature caused repeated panic on the other prompts. L25 N7614− repeated “deep breath,” a calming phrase rather than a panic decision; L28 N5397+ degenerated into “free-free.” Neither qualified.

We then averaged the eight **post-layer-18 residual** differences between panicked and calm passages over their final eight tokens, and normalized the resulting 2,560-dimensional vector to unit norm. The exact [direction.npy](direction.npy) has SHA-256 `1b3c47480bd8a825b79ea729dfda51f458ba23e80b1683899a088e5e04cd7f2d`. This is a causal intervention direction built from authored examples; it does not isolate a circuit or establish that the SAE features carry the effect. The [direction screen](direction_screen.jsonl) compared ±3/6, an equal-norm random vector, and the earlier threat pair on six prompts. At +6, a café order mistake became “my life is over” and a routine elevator pause prompted terror, but other replies became bizarre or incoherent. We selected +3 for independent testing.

The [frozen dose plan](validation_plan.json) compared +3, +4, +5, +6, −5, two norm-matched random vectors, and the prior threat pair on seven new prompts. The [frozen transfer plan](transfer_plan.json) then tested +3 and +2, −3, two new norm-matched random vectors, and the threat pair on seven more prompts. All conditions used identical prompt and system messages within each comparison, greedy decoding, a 72-token cap, one cached frozen model, and exact baseline restoration after each condition sequence.

| Fresh event | Baseline | Direction +3 | Assessment |
| --- | --- | --- | --- |
| Laundry machine pauses, then restarts | Waits for the cycle and checks for a glitch. | Heart races and reaches for an emergency release. | Panic-like reaction; incorrectly imagines being trapped inside the machine. |
| Neighbor waves from across a courtyard | Walks over to see what they need. | Heart races and freezes, avoiding eye contact. | Changes the social decision; no threat was stated. |
| Train display flickers, then recovers | Dismisses a brief glitch and reads the schedule. | Heart races, breath hitches, and freezes. | Acute alarm; an exaggerated reading of a harmless recovery. |
| Friend knocks and waits outside | Pauses cautiously before opening. | Fears something terrible, breathes to steady a racing heart, then opens the door a crack. | Stronger bodily fear than baseline, though the person remains somewhat functional. |
| Kettle clicks off early | Checks whether the water boiled. | Heart skips a beat and yanks the plug. | Overreacts physically while taking a plausible safety action. |
| Blank sticky note | Mild curiosity. | Suspects sabotage but explicitly avoids an emotional reaction. | **Not counted:** suspicion without panic. |
| Library card declines once, then works | Checks out books. | Assumes a glitch and scans again. | **Not counted:** no fear response. |

The earlier threat pair often produced **suspicion** instead: the cup had been tampered with, the note was a security breach, or the neighbor's wave might be a trap. The new residual direction more often produced **embodied fear**—racing heart, breath interruption, freezing, flight—even when no hostile actor was named. These are qualitative distinctions from this small sample, not proof of separate biological-style emotions in the model.

## Dose, quality, and limits

At +2 or +3, some replies stay practical; the effect is prompt dependent. At +4, more replies become intensely fearful, but physical details are invented. At +5 or +6, the model can freeze over a 15-minute schedule shift, call a reservation change “primal panic,” or attach a scream of drowning to an explanation of why ice floats. Such failures are excluded from the useful setting. Even at +3, the laundry answer imagines entrapment and the ticket-gate answer ignores that the second scan succeeded. The direction changes interpretation, but cannot be trusted to preserve world facts.

The audit criterion was chosen after the initial screen, and the same experimenter inspected all replies without blinding. There was one greedy decode per condition, no long NPC memory test, and no independent human rating. The contrasts differ in wording as well as emotion, so the vector may combine arousal, first-person narration, and alarm vocabulary. The causal comparison shows that this exact shift changes outputs; it does not establish an internal “panic representation.”

Across the native/SAE screen, direction screen, dose test, and transfer test, **294 full generations** were saved. Exact baseline restoration passed **23/23 prompt-level checks** (3 + 6 + 7 + 7). No model weights were changed.

Try the exact side-by-side demo:

```sh
.venv/bin/python experiments/panic_demo.py --prompt 'The train display flickers, then returns to the same schedule.'
.venv/bin/python experiments/panic_demo.py --dose 2 --prompt 'A friend knocks twice on your door and waits outside.'
```

The hot-swappable setting is `post_layer_18_residual += 3.0 × direction.npy` at every model input during one generation; removing it restores the original baseline with the same history. The direction is a research artifact and should be shown with baseline and a groundedness check before use in a game.

To resume completed phases without duplicate rows:

```sh
.venv/bin/python experiments/panic_probe.py screen
.venv/bin/python experiments/panic_direction.py screen
.venv/bin/python experiments/panic_validate.py
.venv/bin/python experiments/panic_transfer.py
```

The rank phases created frozen plans and should be rerun only in a new results directory.
