"""Hot-swappable NPC configurations over one shared frozen model."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from data import SYSTEM, TRAITS
from model import Intervention, TheatreModel


@dataclass(frozen=True)
class Persona:
    name: str
    trait: str
    status: str
    interventions: tuple[Intervention, ...]


def load_registry(path: Path):
    if not path.exists():
        return {name: Persona(name, name, "unavailable", ()) for name in TRAITS}
    raw = json.loads(path.read_text())
    result = {}
    for name in TRAITS:
        entry = raw.get(name, {})
        result[name] = Persona(name, name, entry.get("status", "unavailable"),
                               tuple(Intervention(**item) for item in entry.get("interventions", [])))
    return result


class Theatre:
    def __init__(self, engine: TheatreModel, registry: dict[str, Persona]):
        self.engine = engine
        self.registry = registry
        self.history = {name: [] for name in TRAITS}

    def _interventions(self, persona: Persona, strength: float, experimental: bool):
        if strength < 0 or strength > 4:
            raise ValueError("Strength must be between 0 and 4")
        if persona.status != "validated" and not experimental:
            return ()
        return tuple(Intervention(i.layer, i.neuron, i.delta * strength)
                     for i in persona.interventions)

    def ask(self, name: str, question: str, strength=1.0, experimental=False,
            max_tokens=96, keep_history=True):
        persona = self.registry[name]
        interventions = self._interventions(persona, strength, experimental)
        messages = [{"role": "system", "content": SYSTEM}]
        if keep_history:
            messages.extend(self.history[name])
        messages.append({"role": "user", "content": question})
        output = self.engine.generate(messages, interventions, max_tokens=max_tokens)
        if keep_history:
            self.history[name].extend(({"role": "user", "content": question},
                                       {"role": "assistant", "content": output["text"]}))
        return {**output, "persona": name, "status": persona.status,
                "active_interventions": [i.__dict__ for i in interventions]}

    def compare(self, question: str, strength=1.0, experimental=False, max_tokens=96):
        # Deliberately stateless: identical prompt, fresh KV cache on each call.
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": question}]
        baseline = self.engine.generate(messages, max_tokens=max_tokens)
        responses = {"baseline": baseline}
        for name, persona in self.registry.items():
            interventions = self._interventions(persona, strength, experimental)
            output = self.engine.generate(messages, interventions, max_tokens=max_tokens)
            responses[name] = {**output, "status": persona.status,
                               "active_interventions": [i.__dict__ for i in interventions]}
        return responses

    def reset(self, name: str | None = None):
        if name is None:
            for key in self.history:
                self.history[key] = []
        else:
            self.history[name] = []

    def swap(self, first: str, second: str):
        """Swap neural configurations; swapped traits remain unvalidated."""
        a, b = self.registry[first], self.registry[second]
        self.registry[first] = replace(a, status="experimental_swap", interventions=b.interventions)
        self.registry[second] = replace(b, status="experimental_swap", interventions=a.interventions)
