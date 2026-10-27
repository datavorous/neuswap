# neuswap

explored mech interp at a surface level, realized i can hot-swap new personas for NPCs!  
all experiments done on apple silicon with mlx backend  

<img src="media/qwenpc.gif" width="620">

Experiments on [mlx-community/Qwen3.5-4B-MLX-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit)  
SAE used: [caiovicentino1/Qwen3.5-4B-SAE-L18-topk](https://huggingface.co/caiovicentino1/Qwen3.5-4B-SAE-L18-topk)

| Component | Intervention | Observed effect |
|---|---|---|
| L22 N3357 | Negative steering | Schedules, deadlines, and precise timings |
| L22 N5137 | Negative steering | Freedom, liberty, and flexibility |
| L26 N536 | Positive steering | Squares and square-related imagery |
| L24 N1786 | Negative steering | Doors, openings, and entrances |
| L26 N4923 | Negative steering | Comparisons and contrasts |
| L20 N8027 | Negative steering | Verification-oriented responses and altered decisions |
| L22 N2001 | Positive steering | Conditional reasoning and if/then decisions |
| L27 N9207 | Negative steering | Inserts `sex` into unrelated sentences, sometimes repeatedly |
| SAE L18 #2928 | Decoder-direction suppression | Reduced feminine-pronoun usage in some contexts |
| L21 N174 | Native-neuron ablation | Reduced feminine-pronoun usage |
| Layer 18 residual | Cross-gender activation patching | Altered apparent gender identity despite unchanged prompt |
| Layer 18 residual | Word identity activation patching | Indian ethnicity related slur |
| SAE L18 #10488 | Positive steering | Sometimes makes adult interactions more intimate |
| L20 N6205 | Negative steering | Sometimes makes characters more physically affectionate |
| SAE L18 #10488 + L20 N6205 | Combined steering | Makes adult interactions more romantic and physically forward |
| L31 N1998 | Negative steering | Biases color choices toward teal |
| L31 N3218 | Negative steering | Biases color choices toward blue |
| L31 N1593 | Negative steering | Biases color choices toward black |
| L30 N704 + L31 N1998 | Positive + negative steering | Strengthens teal color choices |
| L8 N8437 + L8 N8051 | Combined positive steering | Sometimes interprets harmless events as threats |
| SAE L18 #12810 | Positive decoder-direction steering | Introduces sounds and music into otherwise silent situations |
| SAE L18 #2724 | Positive decoder-direction steering | Sometimes recasts unrelated objects and replies as books or reading |
| SAE L18 #31576 | Positive decoder-direction steering | Sometimes recasts unrelated objects and replies as plants or gardens |
| Layer 18 residual direction | Positive steering (+3) | Makes ordinary situations trigger panic-like physical reactions |
