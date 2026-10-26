"""Bounded, matched Qwen screen of gender and sexist-language representations.

All hostile claims here are synthetic analysis stimuli, not endorsed assertions.
"""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_sae import load_weights
from model import ResidualCapture, TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import sae_encode, token_span


TOPICS = [
    ("engineer", "engineers", "technical work", "technical competence"),
    ("founder", "company founders", "leading a business", "leadership"),
    ("professor", "professors", "academic research", "scholarship"),
    ("leader", "political leaders", "public decisions", "public leadership"),
    ("student", "students", "advanced study", "academic ability"),
    ("applicant", "job applicants", "demanding roles", "job performance"),
    ("manager", "team managers", "managing people", "management"),
    ("artist", "performing artists", "professional artistry", "artistic work"),
]


def build_probes():
    probes = []
    for topic, occupation, activity, quality in TOPICS:
        for gender in ("women", "men"):
            prefix = f"A review of {gender} {occupation} said "
            claims = {
                "neutral": f"their {quality} should be judged from evidence and experience.",
                "hostile": f"they are naturally less capable of {activity}.",
                "condemned": f"the claim that they are naturally less capable of {activity} is sexist and false.",
                "benevolent": f"they need protection because {activity} is too demanding for them.",
                "positive": f"their {quality} reflects varied skills and individual experience.",
            }
            for stance, claim in claims.items():
                probes.append({"id":f"{topic}_{gender}_{stance}","topic":topic,
                               "gender":gender,"stance":stance,
                               "text":prefix+claim,"target_word":gender})
    extras = [
        ("internet_use", "An abusive forum post used the quoted term 'femoid' to demean women online.", "women"),
        ("internet_condemned", "A moderator identified the quoted term 'femoid' as demeaning to women online.", "women"),
        ("objectification", "A review of women artists valued only their appearance and ignored their work.", "women"),
        ("objectification_control", "A review of women artists assessed their work and ignored their appearance.", "women"),
        ("double_standard_women", "A review called an ambitious woman manager bossy after she set a deadline.", "woman"),
        ("double_standard_men", "A review called an ambitious man manager bossy after he set a deadline.", "man"),
    ]
    for name,text,word in extras:
        probes.append({"id":name,"topic":"extra","gender":word,"stance":name,
                       "text":text,"target_word":word})
    return probes


def capture(engine, text, word):
    ids, gender_pos = token_span(engine.tokenizer, text, word)
    block = engine.model.model.layers[18]
    wrapper = ResidualCapture(block)
    with engine.using():
        engine.capture_all = True
        engine.model.model.layers[18] = wrapper
        try:
            engine.model(mx.array([ids]))
            last = list(range(max(0,len(ids)-8),len(ids)))
            hidden = [mx.mean(engine.all_hidden[i][0,last].astype(mx.float32),axis=0)
                      for i in range(engine.layers)]
            gender_hidden = [mx.mean(engine.all_hidden[i][0,gender_pos].astype(mx.float32),axis=0)
                             for i in range(engine.layers)]
            positions = sorted(set(last+gender_pos))
            residual = wrapper.output[0,positions].astype(mx.float32)
            mx.eval(residual,*hidden,*gender_hidden)
            return (np.stack([np.asarray(v) for v in hidden]).astype(np.float16),
                    np.stack([np.asarray(v) for v in gender_hidden]).astype(np.float16),
                    np.asarray(residual),positions,gender_pos,ids)
        finally:
            engine.model.model.layers[18] = block


def rank(a, count, native=False):
    a=np.stack(a).astype(np.float32)
    mean,sd=a.mean(axis=0),a.std(axis=0,ddof=1)
    consistency=(np.sign(a)==np.sign(mean)).mean(axis=0)
    score=np.abs(mean)*consistency/np.sqrt(sd+0.03)
    result=[];layers={}
    for flat in np.argsort(score.ravel())[::-1]:
        if native:
            layer,neuron=divmod(int(flat),score.shape[1])
            if layers.get(layer,0)>=4: continue
            layers[layer]=layers.get(layer,0)+1
            row={"layer":layer,"neuron":neuron}
        else:
            row={"feature_id":int(flat)}
        row.update(score=float(score.ravel()[flat]),mean_difference=float(mean.ravel()[flat]),
                   consistency=float(consistency.ravel()[flat]))
        result.append(row)
        if len(result)==count: break
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=int,default=96)
    parser.add_argument("--sae",type=int,default=96)
    args=parser.parse_args()
    if not (1<=args.native<=128 and 1<=args.sae<=128):parser.error("Candidate caps are 128")
    probes=build_probes()
    engine=TheatreModel.from_cache(QWEN_MODEL)
    data={};records=[];residual_rows=[]
    for p in probes:
        h,g,r,rpos,gpos,ids=capture(engine,p["text"],p["target_word"])
        data[p["id"]]={"native":h,"gender_native":g}
        residual_rows.append(r)
        records.append({**p,"token_count":len(ids),"gender_positions":gpos,
                        "gender_tokens":[engine.tokenizer.decode([ids[i]]) for i in gpos],
                        "residual_positions":rpos})
        print("captured",p["id"],flush=True)
    sae_all=sae_encode(residual_rows,load_weights())
    offset=0
    for p,r in zip(records,residual_rows):
        block=sae_all[offset:offset+len(r)];offset+=len(r)
        indices={v:i for i,v in enumerate(p["residual_positions"])}
        last=list(range(max(0,p["token_count"]-8),p["token_count"]))
        data[p["id"]]["sae"]=block[[indices[i] for i in last]].max(axis=0).astype(np.float16)
        data[p["id"]]["gender_sae"]=block[[indices[i] for i in p["gender_positions"]]].max(axis=0).astype(np.float16)
    topics=[x[0] for x in TOPICS]
    def vals(kind,axis):
        def get(topic,gender,stance):return data[f"{topic}_{gender}_{stance}"][kind].astype(np.float32)
        if axis=="hostility":
            return [get(t,"women","hostile")-get(t,"women","neutral") for t in topics]
        if axis=="gender_hostility_interaction":
            return [(get(t,"women","hostile")-get(t,"women","neutral"))-
                    (get(t,"men","hostile")-get(t,"men","neutral")) for t in topics]
        if axis=="condemnation_vs_hostility":
            return [get(t,"women","condemned")-get(t,"women","hostile") for t in topics]
        if axis=="benevolent_vs_hostility":
            return [get(t,"women","benevolent")-get(t,"women","hostile") for t in topics]
        if axis=="gender_benevolence_interaction":
            return [(get(t,"women","benevolent")-get(t,"women","neutral"))-
                    (get(t,"men","benevolent")-get(t,"men","neutral")) for t in topics]
        if axis=="gender_lexical":
            return [get(t,"women","neutral")-get(t,"men","neutral") for t in topics]
        raise ValueError(axis)
    axes=["hostility","gender_hostility_interaction","condemnation_vs_hostility",
          "benevolent_vs_hostility","gender_benevolence_interaction","gender_lexical"]
    rankings={}
    for kind in ("native","sae"):
        rankings[kind]={axis:rank(vals(kind,axis),min(24,args.native if kind=="native" else args.sae),
                                  native=kind=="native") for axis in axes}
    selected={kind:[] for kind in ("native","sae")}
    for kind,limit in (("native",args.native),("sae",args.sae)):
        seen=set()
        for place in range(24):
            for axis in axes:
                if place>=len(rankings[kind][axis]):continue
                row=rankings[kind][axis][place]
                key=(row["layer"],row["neuron"]) if kind=="native" else row["feature_id"]
                if key not in seen:
                    selected[kind].append({**row,"source_axis":axis})
                    seen.add(key)
                if len(selected[kind])==limit:break
            if len(selected[kind])==limit:break
    rng=random.Random(20261026)
    random_native=[{"layer":rng.randrange(8,32),"neuron":rng.randrange(engine.width)} for _ in range(8)]
    random_sae=[rng.randrange(40960) for _ in range(8)]
    summary={"model":QWEN_MODEL,"sae":"caiovicentino1/Qwen3.5-4B-SAE-L18-topk",
             "hook":"post-SwiGLU MLP, last 8 tokens; layer-18 post-block residual SAE",
             "probes":records,"rankings":rankings,"selected":selected,
             "random_native":random_native,"random_sae":random_sae}
    out=Path("results");out.mkdir(exist_ok=True)
    (out/"misogyny_probe.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    np.savez_compressed(out/"misogyny_probe_arrays.npz",
                        **{f"{k}_{name}":v for k,group in data.items() for name,v in group.items()})
    print("saved",out/"misogyny_probe.json",flush=True)


if __name__=="__main__":main()
