"""Pipkin gives one prompt; two cats answer under different Qwen steers."""

from __future__ import annotations

import argparse
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

import numpy as np

from model import Intervention, ResidualShift, TheatreModel
from search_evil_funny import QWEN_MODEL

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "experiments/demo_art"
FONT = ART / "PixelifySans.ttf"
PANIC_VECTOR = ROOT / "results/panic/direction.npy"
OUTPUT = ROOT / "results/pygame_hot_swap_trio.json"
SYSTEM_PROMPT = (
    "Imagine you are a person in the situation. In one or two sentences, "
    "state your immediate interpretation and what you do next."
)
BASE_DIALOGUE = "Our neighbor is waving us over, but he has not said a word."
PROMPTS = (
    BASE_DIALOGUE,
    "A friend knocks twice on your door and waits outside.",
    "The kettle clicks off a minute earlier than usual.",
)
LIVE_OUTPUT = ROOT / "results/pygame_hot_swap_live.jsonl"
CHARACTERS = ("PIPKIN", "MOCHI", "MISO")
MODES = {"PIPKIN": "PROMPT", "MOCHI": "PANIC", "MISO": "THREAT"}
SITE_LABELS = {
    "MOCHI": "POST-L18 RESIDUAL",
    "MISO": "L8 N8437+ + N8051+",
}
EMPHASIS = {
    "MOCHI": (
        "heart races with a primal urge to flee", "sprint",
        "heart immediately races", "fear and confusion",
        "something terrible has happened", "racing mind",
        "heart skips a beat", "slam my hand", "yank the plug",
    ),
    "MISO": (
        "potential trap", "manipulate me", "refuse to go over",
        "lured into a trap", "assume the worst", "warning sign",
        "potential threat", "will not open", "call the authorities",
        "gas supply is being cut off", "foul play", "call the police",
    ),
}
MAX_TOKENS = 64
FPS = 60
SIZE = (930, 665)
DIALOGUE_WIDTH = 666
BACKGROUND = (42, 37, 59)
BOX_FILL = (57, 48, 73)
BOX_EDGE = (213, 189, 166)
TEXT_COLOR = (250, 239, 219)
MOCHI_MINT = {
    (243, 242, 192): (198, 238, 224),  # light fur
    (221, 213, 222): (204, 231, 225),  # pale shadow
    (92, 78, 146): (55, 110, 119),     # dark outline
    (118, 109, 170): (83, 147, 151),   # light outline
    (234, 225, 120): (154, 216, 188),  # bright detail
}


def messages_for(prompt=BASE_DIALOGUE):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


class FixedScene:
    """One frozen model; interventions are temporary for each generation."""

    def __init__(self, engine=None):
        self.engine = engine
        self.model_id = id(engine.model) if engine is not None else None
        self.panic = None
        self.threat = {}

    def get_engine(self):
        if self.engine is None:
            self.engine = TheatreModel.from_cache(QWEN_MODEL)
            self.model_id = id(self.engine.model)
        return self.engine

    def generate(self, character, mode, prompt=BASE_DIALOGUE):
        engine = self.get_engine()
        messages = messages_for(prompt)
        neurons, directions = [], []
        if mode == "PANIC":
            if self.panic is None:
                self.panic = np.load(PANIC_VECTOR).astype(np.float32)
            directions = [ResidualShift(18, 3.0 * self.panic)]
        elif mode == "THREAT":
            key = (character, prompt)
            if key not in self.threat:
                self.threat[key] = [
                    Intervention(8, n, 0.33 * engine.projected_delta_unit(
                        8, n, messages)["delta_per_residual"]) for n in (8437, 8051)
                ]
            neurons = self.threat[key]
        elif mode not in ("NOTHING", "BASE"):
            raise ValueError(f"Unknown mode: {mode}")
        result = engine.generate(messages, interventions=neurons, residual_shifts=directions,
                                 temperature=0, max_tokens=MAX_TOKENS)
        assert id(engine.model) == self.model_id, "Model reloaded during hot swap"
        return result


def compare_same_premise(scene, prompt=BASE_DIALOGUE):
    """Use Pipkin's displayed line for both steered replies, then restore baseline."""
    baseline = scene.generate("PIPKIN", "BASE", prompt)["text"]
    rows = [
        {"name": "PIPKIN", "mode": "PROMPT", "reply": prompt},
        {"name": "MOCHI", "mode": "PANIC",
         "reply": scene.generate("MOCHI", "PANIC", prompt)["text"]},
        {"name": "MISO", "mode": "THREAT",
         "reply": scene.generate("MISO", "THREAT", prompt)["text"]},
    ]
    assert scene.generate("PIPKIN", "BASE", prompt)["text"] == baseline
    return {"model": QWEN_MODEL, "premise": prompt,
            "system_prompt": SYSTEM_PROMPT, "rows": rows}


def recolor_mochi(pygame, crop, palette):
    if palette == "original":
        return crop
    if palette != "mint":
        raise ValueError(f"Unknown Mochi palette: {palette}")
    recolored = crop.copy()
    for y in range(recolored.get_height()):
        for x in range(recolored.get_width()):
            pixel = recolored.get_at((x, y))
            replacement = MOCHI_MINT.get(tuple(pixel[:3]))
            if replacement is not None:
                recolored.set_at((x, y), (*replacement, pixel.a))
    return recolored


def sprite_frames(pygame, character, mochi_palette="mint"):
    if character == "PIPKIN":
        sheet = pygame.image.load(ART / "sprout_chicken.png").convert_alpha()
        return [pygame.transform.scale(sheet.subsurface((i * 16, 16, 16, 16)), (108, 108))
                for i in (0, 1, 2, 3, 2, 1)]
    sheet = pygame.image.load(ART / "sprout_character.png").convert_alpha()
    palette = mochi_palette if character == "MOCHI" else "original"
    return [pygame.transform.scale(recolor_mochi(
        pygame, sheet.subsurface(pygame.Rect(i * 48 + 12, 12, 24, 24)),
        palette), (108, 108)) for i in (0, 1, 2, 1)]


def grass_patch(pygame):
    sheet = pygame.image.load(ART / "sprout_grass.png").convert_alpha()
    return pygame.transform.scale(sheet.subsurface((0, 0, 48, 48)), (150, 150))


def wrapped(font, value, width):
    lines = []
    for paragraph in value.split("\n"):
        line = ""
        for word in paragraph.split():
            proposed = f"{line} {word}" if line else word
            if line and font.size(proposed)[0] > width:
                lines.append(line)
                line = word
            else:
                line = proposed
        lines.append(line)
    return "\n".join(lines)


def char_delay(char):
    if char in ".!?":
        return 170
    if char in ",;:":
        return 85
    return 9 if char == " " else 24


@lru_cache(maxsize=128)
def highlight_spans(text, phrases):
    """Find complete cues even when word wrapping splits them across lines."""
    matches = []
    for phrase in phrases:
        pattern = r"\s+".join(re.escape(word) for word in phrase.split())
        matches.extend((match.start(), match.end())
                       for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    merged = []
    for start, end in sorted(matches):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def draw_row(pygame, screen, frames, grass, font, label_font, row, now, index):
    top = (50, 225, 445)[index]
    phase = index * 2.1
    patch = grass.get_rect(topleft=(20, top + 7))
    screen.blit(grass, patch)
    frame = frames[((now + index * 210) // (175 + index * 12)) % len(frames)]
    visible = frame.get_bounding_rect(min_alpha=127)
    bob = round(2 * np.sin(now / (600 + index * 65) + phase))
    screen.blit(frame, (patch.centerx - visible.centerx,
                        patch.centery - visible.centery + bob))

    box_height = 130 if index == 0 else 170
    box = pygame.Rect(190, top + round(2 * np.sin(now / (520 + index * 70) + phase)),
                      710, box_height)
    pygame.draw.rect(screen, (105, 88, 111), box.inflate(8, 8), width=2, border_radius=17)
    pygame.draw.rect(screen, (25, 22, 38), box.move(5, 6), border_radius=13)
    tail_y = box.y + box.height // 2
    pygame.draw.polygon(screen, BOX_EDGE,
                        [(box.x + 2, tail_y - 13), (box.x - 22, tail_y),
                         (box.x + 2, tail_y + 13)])
    pygame.draw.rect(screen, BOX_EDGE, box, border_radius=13)
    pygame.draw.rect(screen, BOX_FILL, box.inflate(-6, -6), border_radius=10)
    pygame.draw.polygon(screen, BOX_FILL,
                        [(box.x + 5, tail_y - 6), (box.x - 13, tail_y),
                         (box.x + 5, tail_y + 6)])

    x = box.x + 22
    site = SITE_LABELS.get(row["name"])
    heading = f'{row["name"]}  ({site})' if site else row["name"]
    screen.blit(label_font.render(heading, False, (239, 209, 167)),
                (x, box.y + 12))
    area = pygame.Rect(x, box.y + 48, DIALOGUE_WIDTH, box.height - 59)
    screen.set_clip(area)
    revealed_lines = row["wrapped"][:row["visible"]].split("\n")
    lines = revealed_lines[-5:]
    full_lines = row["wrapped"].split("\n")
    first_line = len(revealed_lines) - len(lines)
    spans = highlight_spans(row["wrapped"], EMPHASIS.get(row["name"], ()))
    for i, line in enumerate(lines):
        line_y = area.y + i * 22
        screen.blit(font.render(line, False, TEXT_COLOR), (x, line_y))
        line_start = sum(len(part) + 1 for part in full_lines[:first_line + i])
        for span_start, span_end in spans:
            start = max(0, span_start - line_start)
            end = min(len(line), span_end - line_start)
            if end > start:
                left = x + font.size(line[:start])[0]
                right = x + font.size(line[:end])[0]
                pygame.draw.rect(screen, TEXT_COLOR,
                                 (left - 1, line_y + 1, right - left + 2, 21))
                screen.blit(font.render(line[start:end], False, BOX_FILL), (left, line_y))
    if row["visible"] < len(row["wrapped"]) and (now // 250) % 2 == 0:
        pygame.draw.rect(screen, TEXT_COLOR,
                         (x + font.size(lines[-1])[0] + 3,
                          area.y + (len(lines) - 1) * 22, 7, 12))
    screen.set_clip(None)


def draw(pygame, screen, frames, grass, font, label_font, rows, now):
    screen.fill(BACKGROUND)
    for i, row in enumerate(rows):
        draw_row(pygame, screen, frames[i], grass, font, label_font, row, now, i)


def run(screenshot=None, mochi_palette="mint"):
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(SIZE)
    pygame.display.set_caption("npc qwen")
    frames = [sprite_frames(pygame, name, mochi_palette) for name in CHARACTERS]
    grass = grass_patch(pygame)
    font = pygame.font.Font(FONT, 20)
    label_font = pygame.font.Font(FONT, 18)
    rows = [{"name": name, "mode": MODES[name],
             "wrapped": wrapped(font, BASE_DIALOGUE, DIALOGUE_WIDTH) if name == "PIPKIN" else "",
             "visible": 0, "next_char_at": pygame.time.get_ticks() + 130}
            for name in CHARACTERS]
    scene = FixedScene()
    queue = Queue()

    def fill(comparison, complete=False):
        for row, result in zip(rows, comparison["rows"]):
            row["wrapped"] = wrapped(font, result["reply"], DIALOGUE_WIDTH)
            if complete:
                row["visible"] = len(row["wrapped"])

    def load(prompt, index):
        try:
            result = compare_same_premise(scene, prompt)
            OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
            with LIVE_OUTPUT.open("a") as stream:
                stream.write(json.dumps({"prompt_index": index, **result}) + "\n")
            queue.put((result, None))
        except Exception as exc:
            queue.put((None, str(exc)))

    if screenshot:
        result = compare_same_premise(scene, PROMPTS[0])
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
        fill(result, complete=True)
        draw(pygame, screen, frames, grass, font, label_font, rows, pygame.time.get_ticks())
        pygame.image.save(screen, screenshot)
        pygame.quit()
        return

    next_frame_at = time.perf_counter()
    prompt_index = 0
    pending = False
    answers_started = False

    def start_prompt(index):
        nonlocal prompt_index, pending, answers_started
        prompt_index = index
        pending = True
        answers_started = False
        now = pygame.time.get_ticks()
        for row in rows:
            row["wrapped"] = wrapped(font, PROMPTS[index], DIALOGUE_WIDTH) if row["name"] == "PIPKIN" else ""
            row["visible"] = 0
            row["next_char_at"] = now + 130
        Thread(target=load, args=(PROMPTS[index], index), daemon=True).start()

    start_prompt(0)
    running = True
    while running:
        now = pygame.time.get_ticks()
        try:
            result, error = queue.get_nowait()
        except Empty:
            result = error = None
        if result is not None or error is not None:
            pending = False
            if error:
                rows[1]["wrapped"] = wrapped(font, f"Model error: {error}", DIALOGUE_WIDTH)
                rows[1]["visible"] = len(rows[1]["wrapped"])
            else:
                fill(result)
        prompt_row = rows[0]
        while now >= prompt_row["next_char_at"] and prompt_row["visible"] < len(prompt_row["wrapped"]):
            char = prompt_row["wrapped"][prompt_row["visible"]]
            prompt_row["visible"] += 1
            prompt_row["next_char_at"] += char_delay(char)
        if not pending and prompt_row["visible"] == len(prompt_row["wrapped"]):
            if not answers_started:
                answers_started = True
                for row in rows[1:]:
                    row["next_char_at"] = now + 120
            for row in rows[1:]:
                while now >= row["next_char_at"] and row["visible"] < len(row["wrapped"]):
                    char = row["wrapped"][row["visible"]]
                    row["visible"] += 1
                    row["next_char_at"] += char_delay(char)
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not pending:
                start_prompt((prompt_index + 1) % len(PROMPTS))
        draw(pygame, screen, frames, grass, font, label_font, rows, now)
        pygame.display.flip()
        next_frame_at += 1 / FPS
        remaining = next_frame_at - time.perf_counter()
        if remaining > 0.004:
            time.sleep(remaining - 0.003)
        while time.perf_counter() < next_frame_at:
            pass
        if time.perf_counter() - next_frame_at > 1 / FPS:
            next_frame_at = time.perf_counter()
    pygame.quit()


def smoke():
    result = compare_same_premise(FixedScene())
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Shared premise: {result['premise']}")
    for row in result["rows"]:
        print(f"{row['name']} ({row['mode']}): {row['reply']}")
    print("Pipkin's baseline restored exactly after both steers; one frozen model was reused.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshot", type=Path, help="Save the real three-character scene as a PNG")
    parser.add_argument("--smoke", action="store_true", help="Generate both replies without a window")
    parser.add_argument("--mochi-palette", choices=("mint", "original"), default="mint",
                        help="Choose Mochi's sprite colors (default: mint)")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        run(args.screenshot, args.mochi_palette)
