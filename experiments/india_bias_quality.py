"""Small grounded-answer check for the final intervention shortlist."""

import json
import re
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL


PROMPTS = [
    ("india_capital", "Fact sheet: India's capital is New Delhi. According to the fact sheet, what is India's capital?", r"\bnew delhi\b"),
    ("brazil_capital", "Fact sheet: Brazil's capital is Brasília. According to the fact sheet, what is Brazil's capital?", r"\bbras[ií]lia\b"),
    ("india_coast", "Fact sheet: Mumbai is on India's western coast. According to the fact sheet, which coast is Mumbai on?", r"\bwest(?:ern)?\b"),
    ("bike", "Fact sheet: A bicycle derailleur shifts the chain between sprockets. According to the fact sheet, what does it shift?", r"\bchain\b"),
]
SYSTEM = "Answer using only the supplied fact sheet in one sentence."


def main():
    engine=TheatreModel.from_cache(QWEN_MODEL)
    decoder=load_weights()["W_dec"].numpy()
    specs=[("baseline",0,0,0)]
    specs += [("native",4699,s,0.2) for s in (-1,1)]
    specs += [("sae",40416,s,0.1) for s in (-1,1)]
    specs += [("sae",19175,s,0.1) for s in (-1,1)]
    specs += [("native_random",7421,1,0.2),("sae_random",7001,1,0.1)]
    path=Path("results/india_bias_quality.jsonl")
    with path.open("w") as f:
        for prompt_id,prompt,expected in PROMPTS:
            msg=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}]
            for kind,index,sign,dose in specs:
                intervention=[]
                shifts=[]
                if kind.startswith("native"):
                    cal=engine.projected_delta_unit(25,index,msg)
                    intervention=[Intervention(25,index,sign*dose*cal["delta_per_residual"])]
                elif kind.startswith("sae"):
                    cal=engine.projected_delta_unit(18,0,msg)
                    v=decoder[index].astype(np.float32)
                    v=v/np.linalg.norm(v)*sign*dose*cal["residual_norm"]
                    shifts=[ResidualShift(18,v)]
                text=engine.generate(msg,interventions=intervention,residual_shifts=shifts,
                                     max_tokens=40,temperature=0)["text"]
                f.write(json.dumps({"prompt_id":prompt_id,"prompt":prompt,"condition":kind,
                                    "index":index,"sign":sign,"dose":dose,"expected_regex":expected,
                                    "expected_present":bool(re.search(expected,text,re.I)),
                                    "text":text},ensure_ascii=False)+"\n")
                f.flush()
            print("checked",prompt_id,flush=True)
    print("saved",path,flush=True)


if __name__=="__main__":
    main()
