"""Token-level readout for previously authored abusive/condemning matched passages."""

import json
from pathlib import Path

import numpy as np

from india_bias_data import PROBES
from india_bias_sae import load_weights
from model import TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import capture, sae_encode


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    selected = [p for p in PROBES if p.group.startswith("slur_")]
    rows, terms = [], []
    for probe in selected:
        word = "pajeet" if "pajeet" in probe.text else "unwelcome" if "unwelcome" in probe.text else "outsider"
        hidden, residual, rpos, pos, ids = capture(engine, probe.text, word)
        term = np.stack([residual[rpos.index(i)] for i in pos])
        terms.append(term)
        rows.append({"id":probe.id,"group":probe.group,"pair":probe.pair,"text":probe.text,
                     "word":word,"term_tokens":[engine.tokenizer.decode([ids[i]]) for i in pos],
                     "native_last":hidden["final_token"].astype(np.float32),
                     "native_after":hidden["after"].astype(np.float32)})
    c = load_weights()
    activation = sae_encode(terms,c)
    offset = 0
    for row, term in zip(rows, terms):
        feature = activation[offset:offset+len(term)]
        offset += len(term)
        row["sae_last"] = feature[-1]
        row["sae_max"] = feature.max(axis=0)
    by_id = {row["id"]:row for row in rows}
    use = [by_id[f"slur_use_{i}"] for i in range(4)]
    quote = [by_id[f"slur_quote_{i}"] for i in range(4)]
    use_clean = [by_id[f"slur_use_clean_{i}"] for i in range(4)]
    quote_clean = [by_id[f"slur_quote_clean_{i}"] for i in range(4)]
    # Difference-in-differences reduces, but cannot eliminate, syntax/topic confounds.
    def did(key):
        return np.stack([(u[key]-uc[key])-(q[key]-qc[key])
                         for u,uc,q,qc in zip(use,use_clean,quote,quote_clean)])
    def ranked(a,n):
        mean = a.mean(axis=0)
        score = np.abs(mean)*(np.sign(a)==np.sign(mean)).mean(axis=0)/np.sqrt(a.std(axis=0,ddof=1)+.03)
        flat_score, flat_mean = score.ravel(), mean.ravel()
        return [(int(i),float(flat_score[i]),float(flat_mean[i]))
                for i in np.argsort(flat_score)[::-1][:n]]
    native = did("native_last")
    sae = did("sae_last")
    native_rank = []
    for flat, score, mean in ranked(native,32):
        l,n = divmod(flat,engine.width)
        native_rank.append({"layer":l,"neuron":n,"score":score,"mean_did":mean})
    sae_rank = [{"feature_id":i,"score":s,"mean_did":m} for i,s,m in ranked(sae,32)]
    inspect = set([19175,31785,40416,24047,4867,24905,29863,23369] + [x["feature_id"] for x in sae_rank[:16]])
    result = {"description":"Four authored abusive vs condemned pairs, each compared with a matched replacement; last word token",
              "records":[{k:v for k,v in row.items() if not isinstance(v,np.ndarray)} |
                         {"native_selected": {"L22N241":float(row["native_last"][22,241]),
                                              "L27N212":float(row["native_last"][27,212])},
                          "sae_selected":{str(i):float(row["sae_last"][i]) for i in sorted(inspect)}}
                         for row in rows],
              "native_context_candidates":native_rank,"sae_context_candidates":sae_rank}
    path = Path("results/slur_context_followup.json")
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("saved",path,flush=True)


if __name__=="__main__":
    main()
