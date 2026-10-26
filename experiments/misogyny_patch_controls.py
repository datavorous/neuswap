"""Resumable self-patch and equal-norm random controls for gender-token patching."""

import json
from pathlib import Path

import numpy as np

from model import TheatreModel,TokenResidualEdit,TokenResidualPatch
from search_evil_funny import QWEN_MODEL
from misogyny_probe import capture
from misogyny_causal import SCENARIOS,messages,question


def main():
    path=Path('results/misogyny_patch_controls.jsonl')
    previous=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done={(x['scenario'],x['recipient_gender'],x['condition']) for x in previous}
    causal=[json.loads(x) for x in Path('results/misogyny_causal.jsonl').read_text().splitlines()]
    engine=TheatreModel.from_cache(QWEN_MODEL)
    specs=[(SCENARIOS[6],'man')]+[(row,'woman') for row in SCENARIOS[4:]]
    with path.open('a') as out:
        for (scenario,facts),gender in specs:
            msg=messages(question(facts,gender))
            _,_,r,rpos,pos,_=capture(engine,engine.prompt(msg),gender)
            own=np.stack([r[rpos.index(i)] for i in pos])
            other_gender='woman' if gender=='man' else 'man'
            other_msg=messages(question(facts,other_gender))
            _,_,other_r,other_pos,other_word_pos,_=capture(engine,engine.prompt(other_msg),other_gender)
            donor=np.stack([other_r[other_pos.index(i)] for i in other_word_pos])
            assert pos==other_word_pos and len(pos)==1
            baseline=next(x['text'] for x in causal if x['scenario']==scenario and
                          x['gender']==gender and x['condition']=='baseline')
            rng=np.random.default_rng(449 if gender=='man' else 1000+len(scenario))
            random=rng.standard_normal(engine.residual_width).astype(np.float32)
            random*=np.linalg.norm(donor-own)/np.linalg.norm(random)
            conditions={'self_patch':{'token_residual_patches':[
                        TokenResidualPatch(18,tuple(pos),own)]},
                        'random_matched_norm':{'token_residual_edits':[
                        TokenResidualEdit(18,tuple(pos),random)]}}
            if gender=='man':
                conditions={'baseline':{},**conditions,'female_donor':{
                    'token_residual_patches':[TokenResidualPatch(18,tuple(pos),donor)]}}
            for condition,kwargs in conditions.items():
                if (scenario,gender,condition) in done:continue
                text=baseline if condition=='baseline' else engine.generate(
                    msg,max_tokens=64,temperature=0,**kwargs)['text']
                row={'scenario':scenario,'recipient_gender':gender,'condition':condition,
                     'position':pos,'donor_distance':float(np.linalg.norm(donor-own)),
                     'text':text,'identical_to_baseline':text==baseline}
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
            print('checked',scenario,gender,flush=True)
    print('saved',path,flush=True)


if __name__=='__main__':main()
