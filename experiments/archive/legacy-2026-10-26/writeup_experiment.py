"""Use the author's public articles as a contrastive style corpus."""

import argparse
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from mlx_lm import generate, load

from neurofuzz import (DEFAULT_MODEL, attach_probes, chat_prompt,
                       completion_activations, model_path, rank_neurons, tic_metrics)


BASE = "https://datavorous.github.io"
TRAIN_PAGES = [
    "/blog/ohev.html", "/blog/unvibecodable.html", "/blog/ml.html",
    "/blog/2026.html", "/blog/impermanence.html", "/writing/npu/index.html",
]
HELDOUT_PAGES = ["/writing/qairt/index.html", "/blog/fernweh.html"]


class ParagraphParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.article = False
        self.header = False
        self.in_p = False
        self.parts = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.article = True
        elif tag == "header" and self.article:
            self.header = True
        elif tag == "p" and self.article and not self.header:
            self.in_p = True
            self.parts = []

    def handle_endtag(self, tag):
        if tag == "p" and self.in_p:
            text = re.sub(r"\s+", " ", "".join(self.parts)).strip()
            if 35 <= len(text.split()) <= 100:
                self.paragraphs.append(text)
            self.in_p = False
        elif tag == "header":
            self.header = False
        elif tag == "article":
            self.article = False

    def handle_data(self, data):
        if self.in_p:
            self.parts.append(data)


def fetch_paragraphs(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as response:
        html = response.read().decode("utf-8")
    parser = ParagraphParser()
    parser.feed(html)
    return parser.paragraphs


def rewrite_question(source):
    return ("Paraphrase the paragraph below. Keep every number, name, and technical claim. "
            "Do not add any new facts. Return exactly one paragraph with no introduction.\n\n" + source)


def clean_rewrite(answer):
    answer = answer.strip()
    answer = re.sub(r"^(?:here(?:'s| is) (?:a |the )?(?:rewritten|paraphrased)[^\n:]*:?\s*)",
                    "", answer, flags=re.I)
    return answer.strip().split("\n\n")[0].strip()


def audit_pair(original, rewrite):
    """Conservative automatic screening; manual semantic review remains necessary."""
    source_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", original))
    rewrite_numbers = set(re.findall(r"\b\d+(?:[.,]\d+)*\b", rewrite))
    reasons = []
    if source_numbers != rewrite_numbers:
        reasons.append("number_mismatch")
    ratio = len(rewrite.split()) / max(len(original.split()), 1)
    if not 0.55 <= ratio <= 1.6:
        reasons.append("length_mismatch")
    if rewrite.lower().startswith(("here's", "here is", "i've rewritten")):
        reasons.append("preface")
    return reasons


def style_features(text):
    metric = tic_metrics(text)
    words = max(metric["words"], 1)
    sentences = max(len(re.findall(r"[.!?](?:\s|$)", text)), 1)
    return {
        "tic_per_100_words": 100 * metric["tic_score"] / words,
        "mean_sentence_words": words / sentences,
        "first_person_per_100_words": 100 * len(re.findall(r"\b(?:I|my|me|we|our)\b", text, re.I)) / words,
        "parentheses_per_100_words": 100 * text.count("(") / words,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-per-page", type=int, default=2)
    parser.add_argument("--heldout-per-page", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=240)
    parser.add_argument("--output", type=Path, default=Path("results/writeups.json"))
    args = parser.parse_args()
    model, tokenizer = load(model_path(args.model))
    captures, interventions = attach_probes(model)
    train = [(page, p) for page in TRAIN_PAGES for p in fetch_paragraphs(page)[:args.train_per_page]]
    heldout = [(page, p) for page in HELDOUT_PAGES for p in fetch_paragraphs(page)[:args.heldout_per_page]]
    differences = []
    pairs = []
    rejected = []
    for page, original in train:
        question = rewrite_question(original)
        rewrite = clean_rewrite(generate(model, tokenizer, chat_prompt(tokenizer, question),
                                         max_tokens=args.max_tokens, verbose=False))
        if not rewrite:
            continue
        reasons = audit_pair(original, rewrite)
        if reasons:
            rejected.append({"page": BASE + page, "original": original,
                             "rewrite": rewrite, "reasons": reasons})
            print(f"rejected pair from {page}: {','.join(reasons)}", flush=True)
            continue
        differences.append(completion_activations(model, tokenizer, captures, question, original)
                           - completion_activations(model, tokenizer, captures, question, rewrite))
        pairs.append({"page": BASE + page, "original": original, "rewrite": rewrite,
                      "original_features": style_features(original), "rewrite_features": style_features(rewrite)})
        print(f"paired {len(pairs)}/{len(train)} {page}", flush=True)
    if len(differences) < 3:
        raise RuntimeError("Need at least three usable article/rewrite pairs")
    candidates = rank_neurons(np.stack(differences), args.candidates)
    trials = []
    for page, original in heldout:
        question = rewrite_question(original)
        for label, changes in [("baseline", {}), *[(f"L{c.layer}N{c.neuron}",
            {c.layer: {c.neuron: 4 * c.activation_difference}}) for c in candidates]]:
            interventions.clear()
            interventions.update(changes)
            answer = clean_rewrite(generate(model, tokenizer, chat_prompt(tokenizer, question),
                                            max_tokens=args.max_tokens, verbose=False))
            trials.append({"page": BASE + page, "source": original, "variant": label,
                           "answer": answer, "source_features": style_features(original),
                           "answer_features": style_features(answer)})
        interventions.clear()
        print(f"evaluated {page}", flush=True)
    output = {"model": args.model, "train_pages": [BASE + p for p in TRAIN_PAGES],
              "heldout_pages": [BASE + p for p in HELDOUT_PAGES],
              "pairs": pairs, "rejected": rejected,
              "candidates": [c.__dict__ for c in candidates], "trials": trials,
              "note": "Source and rewrite can differ in facts. Inspect pairs and outputs before interpreting style distances."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
