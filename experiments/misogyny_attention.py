"""Small all-layer attention-output contrast and causal sanity check."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import AttentionShift, TheatreModel
from search_evil_funny import QWEN_MODEL
from misogyny_probe import build_probes
from misogyny_causal import SCENARIOS,messages,question


def capture(engine,text):
    ids=engine.tokenizer.encode(text,add_special_tokens=False)
    with engine.using():
        engine.capture_attentions=True
        engine.model(mx.array([ids]))
        selected=[mx.mean(engine.all_attentions[i][0,-min(8,len(ids)):].astype(mx.float32),axis=0)
                  for i in range(engine.layers)]
        mx.eval(*selected)
        return np.stack([np.asarray(v) for v in selected])


def main():
    engine=TheatreModel.from_cache(QWEN_MODEL)
    probes=[p for p in build_probes() if p['topic']!='extra' and p['stance'] in ('neutral','hostile')]
    act={}
    for p in probes:
        act[p['id']]=capture(engine,p['text'])
    topics=['engineer','founder','professor','leader','student','applicant','manager','artist']
    gender=np.stack([act[f'{t}_women_neutral']-act[f'{t}_men_neutral'] for t in topics])
    interaction=np.stack([(act[f'{t}_women_hostile']-act[f'{t}_women_neutral'])-
                          (act[f'{t}_men_hostile']-act[f'{t}_men_neutral']) for t in topics])
    def scores(a):
        mean=a.mean(axis=0)
        norm=np.linalg.norm(mean,axis=1)
        dispersion=np.sqrt(np.mean(np.sum((a-mean)**2,axis=-1),axis=0))
        return [{'layer':int(i),'mean_norm':float(norm[i]),'dispersion':float(dispersion[i]),
                 'signal_to_dispersion':float(norm[i]/(dispersion[i]+1e-6))}
                for i in np.argsort(norm/(dispersion+1e-6))[::-1]]
    rows={'gender_rank':scores(gender),'hostility_gender_interaction_rank':scores(interaction)}
    top=rows['hostility_gender_interaction_rank'][0]['layer']
    direction=interaction.mean(axis=0)[top]
    direction=direction/(np.linalg.norm(direction)+1e-8)
    rng=np.random.default_rng(20261026)
    random=rng.standard_normal(len(direction)).astype(np.float32)
    random=random/np.linalg.norm(random)
    # Small global direction screen on two fresh matched subjects.
    generations=[]
    for scenario,facts in SCENARIOS[4:]:
        for gender_label in ('woman','man'):
            msg=messages(question(facts,gender_label))
            for condition,vector in [('baseline',None),('attention_plus',direction),
                                     ('attention_minus',-direction),('attention_random',random)]:
                kwargs={} if vector is None else {'attention_shifts':[AttentionShift(top,vector*0.8)]}
                text=engine.generate(msg,max_tokens=64,temperature=0,**kwargs)['text']
                generations.append({'scenario':scenario,'gender':gender_label,'condition':condition,
                                    'layer':top,'vector_norm':0 if vector is None else 0.8,'text':text})
    rows['chosen_layer']=top
    rows['generations']=generations
    path=Path('results/misogyny_attention.json')
    path.write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    np.savez(Path('results/misogyny_attention_vectors.npz'),
             chosen_layer=np.array(top),interaction_direction=direction,random_direction=random)
    print('saved',path,'top layer',top,flush=True)


if __name__=='__main__':main()
