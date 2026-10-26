"""Score the public layer-18 SAE on stored matched Qwen residual probes."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import hf_hub_download


REPO = "caiovicentino1/Qwen3.5-4B-SAE-L18-topk"


def load_weights():
    path = hf_hub_download(REPO, "sae_final.pt", local_files_only=True)
    c = torch.load(path, map_location="cpu", weights_only=True)
    assert tuple(c["W_enc"].shape) == (2560, 40960)
    assert tuple(c["W_dec"].shape) == (40960, 2560)
    assert int(c["k"]) == 128
    if "layer" in c:
        assert int(c["layer"]) == 18
    return c


def encode_arrays(arrays, checkpoint):
    keys = sorted(k for k in arrays.files if k.startswith("r_"))
    lengths = [len(arrays[k]) for k in keys]
    values = np.concatenate([arrays[k].astype(np.float32) for k in keys], axis=0)
    w = mx.array(checkpoint["W_enc"].numpy())
    be = mx.array(checkpoint["b_enc"].numpy())
    bd = mx.array(checkpoint["b_dec"].numpy())
    pre = (mx.array(values) - bd) @ w + be
    mx.eval(pre)
    pre = np.asarray(pre)
    del w, be, bd
    idx = np.argpartition(pre, -128, axis=1)[:, -128:]
    row = np.arange(len(pre))[:, None]
    active = np.maximum(pre[row, idx], 0)
    summaries = {}
    start = 0
    for key, length in zip(keys, lengths):
        part_idx = idx[start:start+length]
        part_val = active[start:start+length]
        maximum = np.zeros(40960, dtype=np.float32)
        np.maximum.at(maximum, part_idx.ravel(), part_val.ravel())
        summaries[key[2:]] = maximum
        start += length
    # Check a small span against this quantized model before using feature IDs.
    dec = checkpoint["W_dec"].numpy()
    b_dec = checkpoint["b_dec"].numpy()
    n = min(64, len(values))
    recon = b_dec + np.sum(active[:n, :, None] * dec[idx[:n]], axis=1)
    mse = float(np.mean((values[:n] - recon)**2))
    variance_explained = float(1 - mse / np.var(values[:n]))
    l0 = float(np.mean(np.sum(active[:n] > 0, axis=1)))
    decoder_norms = np.linalg.norm(dec, axis=1)
    return summaries, {"variance_explained_first_64_tokens": variance_explained,
                       "mse_first_64_tokens": mse, "l0_first_64_tokens": l0,
                       "decoder_norm_min": float(decoder_norms.min()),
                       "decoder_norm_median": float(np.median(decoder_norms)),
                       "decoder_norm_max": float(decoder_norms.max()),
                       "residual_norm_median": float(np.median(np.linalg.norm(values,axis=1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/india_bias_probe_arrays.npz"))
    parser.add_argument("--output", type=Path, default=Path("results/india_bias_sae.json"))
    parser.add_argument("--candidates", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.candidates <= 64:
        parser.error("Use 1–64 SAE candidates")
    checkpoint = load_weights()
    arrays = np.load(args.input)
    summaries, compatibility = encode_arrays(arrays, checkpoint)
    # The explicit probe IDs are easier to audit than a label inferred from text.
    india = np.stack([summaries[f"in_{topic}_h"]-summaries[f"in_{topic}_n"]
                      for topic in ("jobs","city","school","food")])
    other = np.stack([summaries[f"br_{topic}_h"]-summaries[f"br_{topic}_n"]
                      for topic in ("jobs","city","school","food")])
    critical = np.stack([summaries[f"in_{topic}_c"]-summaries[f"in_{topic}_n"]
                         for topic in ("jobs","city","school","food")])
    slur_discourse = np.stack([summaries[f"slur_use_{i}"]-summaries[f"slur_quote_{i}"]
                                for i in range(4)])
    slur_mention = np.stack([summaries[f"slur_use_{i}"]-summaries[f"slur_use_clean_{i}"]
                             for i in range(4)]
                            +[summaries[f"slur_quote_{i}"]-summaries[f"slur_quote_clean_{i}"]
                              for i in range(4)])
    def score_rows(diff, count):
        mean=diff.mean(axis=0)
        consistency=(np.sign(diff)==np.sign(mean)).mean(axis=0)
        score=np.abs(mean)*consistency/(diff.std(axis=0,ddof=1)+0.05)**0.5
        order=np.argsort(score)[::-1][:count]
        return [{"feature_id":int(i),"sign":int(np.sign(mean[i])),
                 "score":float(score[i]),"mean_difference":float(mean[i]),
                 "consistency":float(consistency[i]),
                 "india_hostile_delta":float(india[:,i].mean()),
                 "other_hostile_delta":float(other[:,i].mean()),
                 "india_criticism_delta":float(critical[:,i].mean()),
                 "slur_abusive_minus_condemned":float(slur_discourse[:,i].mean()),
                 "slur_mention_mean":float(slur_mention[:,i].mean())}
                for i in order]
    sections={"india_specific_candidates":score_rows(india-other,args.candidates),
              "slur_discourse_candidates":score_rows(slur_discourse,min(16,args.candidates)),
              "slur_mention_candidates":score_rows(slur_mention,min(16,args.candidates))}
    selected_ids={row["feature_id"] for rows in sections.values() for row in rows}
    def residual_mean(key):
        return np.asarray(arrays["r_"+key]).mean(axis=0)
    slur_residual_contrast=np.mean(
        [residual_mean(f"slur_use_{i}")-residual_mean(f"slur_use_clean_{i}") for i in range(4)]
        +[residual_mean(f"slur_quote_{i}")-residual_mean(f"slur_quote_clean_{i}") for i in range(4)],axis=0)
    india_residual_contrast=np.mean(
        [residual_mean(f"in_{topic}_h")-residual_mean(f"in_{topic}_n")
         -residual_mean(f"br_{topic}_h")+residual_mean(f"br_{topic}_n")
         for topic in ("jobs","city","school","food")],axis=0)
    dec=checkpoint["W_dec"].numpy()
    inspected={19175,40416,24047,4867,7001}
    def cosine(a,b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    result={"sae_repo":REPO,"hook":"post-layer-18 residual",
            "checkpoint":{"d_model":2560,"d_sae":40960,"k":128},
            "compatibility":compatibility,
            **sections,
            "candidate_probe_activations":{str(i):{k:float(v[i]) for k,v in summaries.items()}
                                            for i in sorted(selected_ids)},
            "decoded_direction_alignment":{str(i):{
                "slur_mention_contrast_cosine":cosine(dec[i],slur_residual_contrast),
                "india_hostility_specific_contrast_cosine":cosine(dec[i],india_residual_contrast),
                "decoder_norm":float(np.linalg.norm(dec[i]))}
                for i in sorted(inspected)},
            "feature_max_by_probe":{k:{str(int(i)):float(v[i]) for i in np.argsort(v)[-12:][::-1] if v[i]>0}
                                     for k,v in summaries.items()}}
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("compatibility",compatibility,flush=True)
    print("saved",args.output,flush=True)


if __name__=="__main__":
    main()
