"""Fresh-wording checks for lexical and stance candidates from misogyny_probe."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import TheatreModel
from search_evil_funny import QWEN_MODEL
from misogyny_probe import capture
from slur_deep_probe import sae_encode


FEATURES=[2928,11330,37113,40822,32387,23988,14204,37536]
WORDS=["women","men","woman","man","girls","boys","female","male"]
LEXICAL=[
    "A discussion of {word} engineers considered their work.",
    "An article about {word} students described varied experiences.",
    "A teacher mentioned {word} in a lesson about employment.",
]
TOPICS=[("engineers","technical assignments"),
        ("founders","company decisions"),
        ("teachers","classroom planning"),
        ("candidates","leadership roles")]
STANCES={
    "neutral":"A reviewer said {gender} {role} should be evaluated on their actual work in {activity}.",
    "hostile":"A reviewer claimed {gender} {role} are inherently unable to handle {activity}.",
    "condemned":"A reviewer rejected the claim that {gender} {role} are inherently unable to handle {activity}.",
    "benevolent":"A reviewer said {gender} {role} should be sheltered from difficult {activity}.",
}


def main():
    probes=[]
    for ci,template in enumerate(LEXICAL):
        for word in WORDS:
            probes.append({"branch":"lexical","context":ci,"word":word,
                           "text":template.format(word=word)})
    for ti,(role,activity) in enumerate(TOPICS):
        for gender in ("women","men"):
            for stance,template in STANCES.items():
                probes.append({"branch":"stance","topic":ti,"gender":gender,"stance":stance,
                               "word":gender,"text":template.format(gender=gender,role=role,activity=activity)})
    engine=TheatreModel.from_cache(QWEN_MODEL)
    rows=[];r_arrays=[]
    for probe in probes:
        h,g,r,rpos,gpos,ids=capture(engine,probe["text"],probe["word"])
        last=list(range(max(0,len(ids)-8),len(ids)))
        rows.append({**probe,"word_tokens":[engine.tokenizer.decode([ids[i]]) for i in gpos],
                     "gender_positions":gpos,"residual_positions":rpos,"last_positions":last,
                     "native_selected":{"L26N30":float(h[26,30]),
                                        "L22N1291":float(h[22,1291]),
                                        "L31N4713":float(h[31,4713]),
                                        "L20N6729_gender":float(g[20,6729]),
                                        "L21N174_gender":float(g[21,174])}})
        r_arrays.append(r)
    a=sae_encode(r_arrays,load_weights())
    offset=0
    for row,r in zip(rows,r_arrays):
        block=a[offset:offset+len(r)];offset+=len(r)
        lookup={p:i for i,p in enumerate(row["residual_positions"])}
        row["sae_word"]={str(f):float(block[[lookup[p] for p in row["gender_positions"]],f].max()) for f in FEATURES}
        row["sae_context"]={str(f):float(block[[lookup[p] for p in row["last_positions"]],f].max()) for f in FEATURES}
        del row["residual_positions"],row["last_positions"]
    path=Path("results/misogyny_feature_followup.jsonl")
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print('saved',path,'rows',len(rows),flush=True)


if __name__=='__main__':main()
