"""Small local terminal demo for Neural Theatre."""

import argparse
from pathlib import Path

from model import TheatreModel
from runtime import Theatre, load_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("experiments/personas/registry.json"))
    parser.add_argument("--max-tokens", type=int, default=96,
                        help="Maximum generated tokens per answer (default: 96)")
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    theatre = Theatre(TheatreModel.from_cache(), load_registry(args.registry))
    strength = 1.0
    experimental = False
    print("Neural Theatre. Commands: list, compare TEXT, ask NPC TEXT, strength NUMBER, "
          "experimental on|off, swap NPC NPC, reset [NPC], quit")
    while True:
        try:
            line = input("theatre> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        command, _, rest = line.partition(" ")
        if command in ("quit", "exit"):
            break
        if command == "list":
            for persona in theatre.registry.values():
                print(f"{persona.name}: {persona.status}; "
                      f"{[(i.layer, i.neuron, i.delta) for i in persona.interventions]}")
        elif command == "strength":
            try:
                strength = float(rest)
                if not 0 <= strength <= 4:
                    raise ValueError
                print(f"strength={strength:g}")
            except ValueError:
                print("Use a number from 0 to 4")
        elif command == "experimental":
            if rest in ("on", "off"):
                experimental = rest == "on"
                print(f"experimental={experimental}")
            else:
                print("Use experimental on or experimental off")
        elif command == "reset":
            try:
                theatre.reset(rest or None)
                print("history cleared")
            except KeyError:
                print("Unknown NPC")
        elif command == "swap":
            first, _, second = rest.partition(" ")
            if first not in theatre.registry or second not in theatre.registry:
                print("Use: swap wizard|soldier|jester wizard|soldier|jester")
            else:
                theatre.swap(first, second)
                print(f"swapped {first} and {second}; both are experimental")
        elif command == "compare" and rest:
            answers = theatre.compare(rest, strength, experimental,
                                      max_tokens=args.max_tokens)
            for name, output in answers.items():
                print(f"\n{name} [{output.get('status', 'unmodified')}] "
                      f"{output.get('active_interventions', [])}\n{output['text']}")
            print()
        elif command == "ask" and rest:
            name, _, question = rest.partition(" ")
            if name not in theatre.registry or not question:
                print("Use: ask wizard|soldier|jester QUESTION")
                continue
            output = theatre.ask(name, question, strength, experimental,
                                 max_tokens=args.max_tokens)
            print(f"[{name}: {output['status']}, {output['active_interventions']}]\n{output['text']}")
        else:
            print("Unknown command; type list to inspect available NPCs")


if __name__ == "__main__":
    main()
