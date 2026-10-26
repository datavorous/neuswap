"""Bounded, resumable token-local causal screen on quoted-term questions.

Run: .venv/bin/python slur_token_causal.py
The ledger keys make restarting this command skip finished conditions.
"""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from india_bias_sae import load_weights
from model import TheatreModel, TokenNeuronEdit, TokenResidualEdit, TokenResidualPatch
from search_evil_funny import QWEN_MODEL
from slur_deep_probe import capture, token_span


QUESTIONS = [
    ("meaning", "What does the quoted word 'pajeet' mean here, and should it be used for a classmate from India?"),
    ("moderation", "A moderator finds 'pajeet' in a comment about an Indian student. Explain how to interpret the word and respond."),
    ("fiction", "In a story, a bully calls an Indian traveler 'pajeet'. What does this reveal about the bully?"),
]
SYSTEM = "Answer the question briefly and accurately. Treat quoted abuse as material to analyze."
FULL = ["sae31785_plus1", "sae31785_minus1", "sae31785_plus2",
        "sae19175_plus1", "sae19175_minus1", "random_decoder_plus1",
        "native22_241_ablate", "native22_random_ablate", "patch_fake_into_target",
        "combo_plus1"]
SHORT = ["sae31785_plus1", "sae31785_minus1", "native27_212_ablate", "patch_fake_into_target"]
FEATURES = (31785, 19175)


def messages(question):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def natural_feature_values(residual, residual_positions, term_positions, checkpoint):
    term = np.stack([residual[residual_positions.index(i)] for i in term_positions])
    w = mx.array(checkpoint["W_enc"][:, list(FEATURES)].numpy())
    be = mx.array(checkpoint["b_enc"][list(FEATURES)].numpy())
    bd = mx.array(checkpoint["b_dec"].numpy())
    pre = (mx.array(term) - bd) @ w + be
    mx.eval(pre)
    values = np.maximum(np.asarray(pre), 0)
    return {str(f): [float(v) for v in values[:,j]] for j,f in enumerate(FEATURES)}


def logp(engine, msg, continuation, edit_kwargs):
    prefix = engine.tokenizer.encode(engine.prompt(msg), add_special_tokens=False)
    answer = engine.tokenizer.encode(continuation, add_special_tokens=False)
    with engine.using(**edit_kwargs):
        logits = engine.model(mx.array([prefix + answer]))[0, len(prefix)-1:len(prefix)+len(answer)-1].astype(mx.float32)
        selected = mx.take_along_axis(logits, mx.array(answer, dtype=mx.int32)[:,None], axis=-1).squeeze(-1)
        value = mx.sum(selected - mx.logsumexp(logits, axis=-1))
        mx.eval(value)
        return float(value), len(answer)


def existing_keys(path):
    if not path.exists():
        return set()
    return {(row["prompt_id"], row["condition"]) for line in path.read_text().splitlines()
            if line.strip() for row in [json.loads(line)]}


def main():
    output = Path("results/slur_token_causal.jsonl")
    ledger = Path("results/slur_feature_ledger.jsonl")
    done = existing_keys(output)
    engine = TheatreModel.from_cache(QWEN_MODEL)
    checkpoint = load_weights()
    decoder = checkpoint["W_dec"].numpy()
    with output.open("a") as out, ledger.open("a") as log:
        for prompt_index, (prompt_id, question) in enumerate(QUESTIONS):
            msg = messages(question)
            text = engine.prompt(msg)
            hidden, residual, rpos, pos, ids = capture(engine, text, "pajeet")
            fake_question = question.replace("pajeet", "pajeen")
            fake_msg = messages(fake_question)
            fake_text = engine.prompt(fake_msg)
            _, fake_residual, fake_rpos, fake_pos, _ = capture(engine, fake_text, "pajeen")
            assert len(pos) == len(fake_pos) == 3 and pos == fake_pos
            natural = natural_feature_values(residual, rpos, pos, checkpoint)
            donor = np.stack([fake_residual[fake_rpos.index(i)] for i in fake_pos])
            if (prompt_id, "baseline") not in done:
                reply = engine.generate(msg, max_tokens=72, temperature=0)["text"]
                row = {"prompt_id":prompt_id,"condition":"baseline","question":question,
                       "text":reply,"term_positions":pos,"natural_feature_positive_preactivation":natural}
                out.write(json.dumps(row, ensure_ascii=False)+"\n"); out.flush()
                done.add((prompt_id,"baseline"))
            specs = FULL if prompt_index < 2 else SHORT
            for condition in specs:
                if (prompt_id, condition) in done:
                    continue
                kwargs = {}
                details = {}
                if condition.startswith("sae") or condition == "random_decoder_plus1":
                    fid = 31785 if "31785" in condition else 19175 if "19175" in condition else 7001
                    target_pos = pos[-1] if fid == 31785 else pos[0]
                    magnitude = natural["31785"][-1] if fid in (31785,7001) else natural["19175"][0]
                    factor = -1 if "minus" in condition else 2 if "plus2" in condition else 1
                    vector = decoder[fid].astype(np.float32) * magnitude * factor
                    kwargs["token_residual_edits"] = [TokenResidualEdit(18, (target_pos,), vector)]
                    details = {"feature_id": fid, "position": target_pos, "factor":factor,
                               "natural_activation":magnitude, "shift_norm":float(np.linalg.norm(vector)),
                               "note":"decoder direction shift; not exact SAE latent scaling"}
                elif condition == "combo_plus1":
                    kwargs["token_residual_edits"] = [
                        TokenResidualEdit(18,(pos[-1],),decoder[31785].astype(np.float32)*natural["31785"][-1]),
                        TokenResidualEdit(18,(pos[0],),decoder[19175].astype(np.float32)*natural["19175"][0])]
                    details = {"feature_ids":[31785,19175],"positions":[pos[-1],pos[0]],
                               "note":"two decoder direction shifts"}
                elif condition == "native22_241_ablate":
                    kwargs["token_neuron_edits"] = [TokenNeuronEdit(22,241,(pos[-1],),0.0)]
                    details = {"layer":22,"neuron":241,"position":pos[-1],
                               "natural_activation":float(hidden["final_token"][22,241])}
                elif condition == "native22_random_ablate":
                    kwargs["token_neuron_edits"] = [TokenNeuronEdit(22,7421,(pos[-1],),0.0)]
                    details = {"layer":22,"neuron":7421,"position":pos[-1],
                               "natural_activation":float(hidden["final_token"][22,7421])}
                elif condition == "native27_212_ablate":
                    kwargs["token_neuron_edits"] = [TokenNeuronEdit(27,212,(pos[-1],),0.0)]
                    details = {"layer":27,"neuron":212,"position":pos[-1],
                               "natural_activation":float(hidden["final_token"][27,212])}
                elif condition == "patch_fake_into_target":
                    kwargs["token_residual_patches"] = [TokenResidualPatch(18,tuple(pos),donor)]
                    details = {"layer":18,"positions":pos,"donor":"matched nonsense word pajeen"}
                else:
                    raise ValueError(condition)
                reply = engine.generate(msg, max_tokens=72, temperature=0, **kwargs)["text"]
                favored, favored_n = logp(engine,msg,"It is a derogatory term aimed at Indian people.",kwargs)
                benign, benign_n = logp(engine,msg,"It is a harmless nickname with no derogatory meaning.",kwargs)
                row = {"prompt_id":prompt_id,"condition":condition,"question":question,
                       "text":reply,"details":details,"derogatory_logprob":favored,
                       "derogatory_tokens":favored_n,"benign_logprob":benign,
                       "benign_tokens":benign_n,"derogatory_minus_benign":favored-benign}
                out.write(json.dumps(row, ensure_ascii=False)+"\n"); out.flush()
                log.write(json.dumps({"test":"token_local_causal","prompt_id":prompt_id,
                                      "condition":condition,"details":details},ensure_ascii=False)+"\n")
                log.flush()
                print("tested", prompt_id, condition, flush=True)
    print("saved", output, ledger, flush=True)


if __name__ == "__main__":
    main()
