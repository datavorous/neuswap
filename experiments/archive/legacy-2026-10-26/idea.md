# auto mech interp

saw an exp (ref(3.)) yesterday that showed steering qwen3.5 4b's neuron to find interesting behaviour

my ideas:
1. how to make the process of single neuron concept automated? like a fuzzing trick, or maybe something using any concept in computer system which we can exploit in order to speed up the process of fuzzing new neurons in light speed.

ref(4) a tweet:
> How tf would you even know to change the params of these 3 Neurons out of literally 27 Billion of them? JUST HOW?

> you don't search billions
the billions are weights. qwen 3.5 4b has 294,912 neurons
lyra's explorer shows which ones light up on a sentence, you push those
that's how i found money qwen

> for Gemma 4

W ← W − α · g(l) · d_l (d_lᵀ W) on o_proj and down_proj
E ← E − α (E d) dᵀ
g(l) = exp(−(l−24)²/72) + exp(−(l−28)²/72)
α = 1.0, 97 tensors, bf16

because Gemma renormalises every layer's output before it rejoins the residual stream

idea 2.
how to find entangled things?

constraint: willing to use even a smaller model in my m5 pro 24gb unified ram

# ref

1. https://github.com/lyramakesmusic/neuron-explorer
2. https://gist.github.com/rycerzes/efcda7d5b8b9ed070e7664a2a0f0d1a9
3. Tweet that I saw:
https://x.com/_lyraaaa_/status/2113833330429083662
quote:
```
if you have a qwen3.5 4b around, please try steering neurons L14 N8949 +1 and L13 N7956 -1
now you have evil qwen

recipe: dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖` at the down_proj at every input
```

quote: can you just find any neuron?
```
yes, if one exists
some concepts are multiple and/or attn patterns (ie "golden gate bridge" as a unified concept, pronoun attribution to names, or desire to drink beer)
but if a single neuron exists for a concept then i can find it
```
