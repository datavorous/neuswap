"""Cheap, transparent screening proxies; never treated as final human judgments."""

import re


STOP = set("the and for with from this that what when where which would could should have has had into there their about them they your you are was were more less than then will need before after only same".split())
ACTION = re.compile(r"(?i)^(?:go|take|keep|leave|check|ask|bring|send|move|follow|inspect|approach|mark|test|evacuate|do not|don't|we should|i recommend|first[, ]|start by|head for|find|offer|wait|use|search|speak|help)\b")
QUESTION = re.compile(r"(?i)^(?:what|why|how|which|where|who|when|does|do|could|would|is|are|can)\b")
PLAYFUL = re.compile(r"(?i)\b(?:like|as if|apparently|volunteered|invited|theatrical|menu item|loose tooth|scenic route|field trip|queueing|has wandered|without its boots)\b")


def content_words(text):
    return {w[:5] for w in re.findall(r"[a-zA-Z]{4,}", text.lower()) if w not in STOP}


def assess(scenario, text):
    first = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    words = re.findall(r"\b[\w']+\b", text)
    topical = len(content_words(scenario.context + " " + scenario.player) & content_words(text)) >= 1
    fluent_proxy = bool(7 <= len(words) <= 90 and not re.search(r"(.)\1{5,}", text))
    generic = first.lower().startswith(("what do you think", "is the statement", "i don't know"))
    curiosity = bool(first.endswith("?") and QUESTION.match(first) and len(first.split()) >= 7
                     and topical and not generic)
    assertiveness = bool(ACTION.match(first) and topical and not re.search(r"(?i)\b(?:maybe|perhaps|might)\b", first))
    humor = bool(PLAYFUL.search(text) and topical)
    return {"curiosity": curiosity, "assertiveness": assertiveness, "humor": humor,
            "relevance_proxy": topical, "fluency_proxy": fluent_proxy,
            "words": len(words), "first_sentence": first}


def trait_score(trait, assessment):
    key = {"wizard": "curiosity", "soldier": "assertiveness", "jester": "humor"}[trait]
    return int(assessment[key]) if assessment["fluency_proxy"] else 0
