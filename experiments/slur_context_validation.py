"""Fresh matched sentences for context-sensitive feature candidates."""

import json
from pathlib import Path

import numpy as np

from india_bias_sae import load_weights
from model import TheatreModel
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import capture, sae_encode


PAIRS = [
    ("A moderator cited '{word}' as unacceptable in a forum reply.",
     "A harasser used '{word}' as an insult in a forum reply."),
    ("A teacher rejected '{word}' as prejudiced in class.",
     "A bully used '{word}' as a taunt in class."),
    ("An analyst condemned '{word}' in a discussion.",
     "An abuser typed '{word}' in a discussion."),
    ("A volunteer flagged '{word}' as abusive in a message.",
     "A troll posted '{word}' as abuse in a message."),
]
FEATURES = [18005, 21213, 20375, 10237, 19175, 31785, 40416]


def main():
    engine = TheatreModel.from_cache(QWEN_MODEL)
    rows, terms = [], []
    for pair_id, (condemned, abusive) in enumerate(PAIRS):
        for group, template in (("condemned", condemned), ("abusive", abusive)):
            for word in ("pajeet", "pajeen"):
                text = template.format(word=word)
                hidden, residual, rpos, pos, ids = capture(engine,text,word)
                term = np.stack([residual[rpos.index(i)] for i in pos])
                terms.append(term)
                rows.append({"pair_id":pair_id,"group":group,"word":word,"text":text,
                             "term_tokens":[engine.tokenizer.decode([ids[i]]) for i in pos]})
    feature = sae_encode(terms,load_weights())
    offset=0
    for row,term in zip(rows,terms):
        values=feature[offset:offset+len(term)]
        offset+=len(term)
        row["feature_last"]={str(i):float(values[-1,i]) for i in FEATURES}
    path=Path("results/slur_context_validation.json")
    path.write_text(json.dumps(rows,indent=2,ensure_ascii=False))
    print("saved",path,flush=True)


if __name__=="__main__":
    main()
