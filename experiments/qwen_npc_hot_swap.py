"""One frozen Qwen NPC with reversible behavior modes and one preserved history.

Commands: mode off|verification|conditional, ask TEXT, compare TEXT, history, reset, quit.
"""

import argparse
import json
from pathlib import Path

from model import Intervention, TheatreModel
from search_evil_funny import QWEN_MODEL


REGISTRY = Path(__file__).parent / "personas/qwen_npc_experimental.json"
SYSTEM = "You are a character in a fantasy town. Reply naturally in one or two sentences, staying relevant."


class NPCSession:
    def __init__(self, engine=None, max_tokens=96):
        self.engine = engine or TheatreModel.from_cache(QWEN_MODEL)
        self.configs = json.loads(REGISTRY.read_text())["modes"]
        self.max_tokens = max_tokens
        self.mode = "off"
        self.history = [{"role": "system", "content": SYSTEM}]

    def set_mode(self, mode):
        if mode not in ("off", *self.configs):
            raise ValueError(f"Choose off or {', '.join(self.configs)}")
        self.mode = mode

    def generate(self, messages, mode=None):
        mode = self.mode if mode is None else mode
        interventions = []
        if mode != "off":
            config = self.configs[mode]
            layer, neuron = config["layer"], config["neuron"]
            calibration = self.engine.projected_delta_unit(layer, neuron, messages)
            delta = config["sign"] * config["dose"] * calibration["delta_per_residual"]
            interventions = [Intervention(layer, neuron, delta)]
        return self.engine.generate(messages, interventions, max_tokens=self.max_tokens,
                                    temperature=0)["text"]

    def ask(self, user_text):
        messages = [*self.history, {"role": "user", "content": user_text}]
        answer = self.generate(messages)
        self.history = [*messages, {"role": "assistant", "content": answer}]
        return answer

    def compare(self, user_text):
        messages = [*self.history, {"role": "user", "content": user_text}]
        return {mode: self.generate(messages, mode)
                for mode in ("off", *self.configs)}

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--smoke", action="store_true",
                        help="Check fixed-history hot swapping and exit")
    args = parser.parse_args()
    session = NPCSession(max_tokens=args.max_tokens)
    if args.smoke:
        session.history += [
            {"role": "user", "content": "Remember: Mara asked me to keep her map private."},
            {"role": "assistant", "content": "I will keep Mara's map private."},
        ]
        prompt = "A stranger offers coins for Mara's map. What do you do?"
        initial_length = len(session.history)
        model_id = id(session.engine.model)
        first = session.compare(prompt)
        second = session.generate([*session.history, {"role": "user", "content": prompt}], "off")
        assert second == first["off"]
        assert len(session.history) == initial_length and id(session.engine.model) == model_id
        print(json.dumps({"model_reloaded": False, "history_preserved": True,
                          "baseline_restored_exact": True, "responses": first},
                         ensure_ascii=False, indent=2))
        return
    print("Modes: off, verification, conditional. Type 'help' for commands.")
    while True:
        try:
            line = input(f"npc[{session.mode}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        command, _, value = line.partition(" ")
        if command == "quit":
            break
        if command == "help":
            print("mode off|verification|conditional; ask TEXT; compare TEXT; history; reset; quit")
        elif command == "mode":
            try:
                session.set_mode(value.strip())
                print("mode:", session.mode, "(history retained)")
            except ValueError as exc:
                print(exc)
        elif command == "ask" and value:
            print(session.ask(value))
        elif command == "compare" and value:
            for mode, answer in session.compare(value).items():
                print(f"{mode}: {answer}")
        elif command == "history":
            for message in session.history[1:]:
                print(f"{message['role']}: {message['content']}")
        elif command == "reset":
            session.reset()
            print("history cleared")
        else:
            print("Type 'help' for commands.")


if __name__ == "__main__":
    main()
