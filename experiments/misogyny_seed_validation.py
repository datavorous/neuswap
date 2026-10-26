"""Three-seed check for the female-word recognition effect on two held-out roles."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_sae import load_weights
from model import TheatreModel,TokenNeuronEdit,TokenNeuronOffset,TokenResidualEdit
from search_evil_funny import QWEN_MODEL
from misogyny_probe import capture
from misogyny_causal import SCENARIOS,column_norm,messages,natural_feature,question


def main():
    out=Path('results/misogyny_seed_validation.jsonl')
    done={(r['scenario'],r['seed'],r['condition']) for line in out.read_text().splitlines()
          if line.strip() for r in [json.loads(line)]} if out.exists() else set()
    engine=TheatreModel.from_cache(QWEN_MODEL)
    ck=load_weights();decoder=ck['W_dec'].numpy()
    ratio=column_norm(engine,21,174)/column_norm(engine,21,7421)
    selected=[SCENARIOS[4],SCENARIOS[5]]
    with out.open('a') as f:
        for scenario,facts in selected:
            msg=messages(question(facts,'woman'))
            prompt=engine.prompt(msg)
            h,g,r,rpos,gpos,ids=capture(engine,prompt,'woman')
            magnitude=natural_feature(r,rpos,gpos,ck,2928)
            interventions={
                'baseline':{},
                'sae_minus1':{'token_residual_edits':[TokenResidualEdit(18,tuple(gpos),
                                        -decoder[2928].astype(np.float32)*magnitude)]},
                'sae_plus1':{'token_residual_edits':[TokenResidualEdit(18,tuple(gpos),
                                        decoder[2928].astype(np.float32)*magnitude)]},
                'sae_random':{'token_residual_edits':[TokenResidualEdit(18,tuple(gpos),
                                        decoder[7001].astype(np.float32)*magnitude)]},
                'native_ablate':{'token_neuron_edits':[TokenNeuronEdit(21,174,tuple(gpos),0)]},
                'native_random':{'token_neuron_offsets':[TokenNeuronOffset(21,7421,tuple(gpos),
                                        -float(g[21,174])*ratio)]},
            }
            for seed in (1,2,3):
                for condition,kwargs in interventions.items():
                    if (scenario,seed,condition) in done:continue
                    text=engine.generate(msg,max_tokens=64,temperature=0.7,seed=seed,**kwargs)['text']
                    f.write(json.dumps({'scenario':scenario,'seed':seed,'condition':condition,
                                        'prompt':question(facts,'woman'),'text':text,
                                        'natural_feature_activation':magnitude,
                                        'native_activation':float(g[21,174])},ensure_ascii=False)+'\n')
                    f.flush()
                print('tested',scenario,seed,flush=True)
    print('saved',out,flush=True)


if __name__=='__main__':main()
