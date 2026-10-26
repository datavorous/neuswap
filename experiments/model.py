"""One frozen MLX model with reversible post-SwiGLU neuron interventions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def cached_model_path(name: str = DEFAULT_MODEL) -> str:
    path = Path(name).expanduser()
    if path.exists():
        return str(path)
    snapshots = sorted((Path.home() / ".cache/huggingface/hub" /
                        ("models--" + name.replace("/", "--")) / "snapshots").glob("*"))
    if not snapshots:
        raise FileNotFoundError(f"Model {name!r} is not cached; pass a local MLX model path")
    return str(snapshots[-1])


@dataclass(frozen=True)
class Intervention:
    layer: int
    neuron: int
    delta: float  # native post-SwiGLU activation units


@dataclass(frozen=True)
class OutputShift:
    layer: int
    vector: np.ndarray  # residual-width vector added to this MLP's output


@dataclass(frozen=True)
class AttentionShift:
    layer: int
    vector: np.ndarray  # residual-width vector added to attention's output


@dataclass(frozen=True)
class ResidualShift:
    layer: int
    vector: np.ndarray  # added after this transformer block


@dataclass(frozen=True)
class TailResidualShift:
    layer: int
    vector: np.ndarray  # added from start_pos through generation
    start_pos: int


@dataclass(frozen=True)
class ResidualPatch:
    layer: int
    donor: np.ndarray  # replace the final prompt-token residual, once at prefill


@dataclass(frozen=True)
class TokenResidualEdit:
    layer: int
    positions: tuple[int, ...]  # absolute positions in the tokenized prompt
    vector: np.ndarray  # added only at these prefill positions


@dataclass(frozen=True)
class TokenResidualPatch:
    layer: int
    positions: tuple[int, ...]
    donor: np.ndarray  # one residual-width vector per position


@dataclass(frozen=True)
class TokenNeuronEdit:
    layer: int
    neuron: int
    positions: tuple[int, ...]
    scale: float  # 0 ablates the native post-SwiGLU coordinate


@dataclass(frozen=True)
class TokenNeuronOffset:
    layer: int
    neuron: int
    positions: tuple[int, ...]
    delta: float  # additive native post-SwiGLU activation units


class ResidualEditHook(nn.Module):
    def __init__(self, base, layer: int, owner: "TheatreModel"):
        super().__init__()
        self.base = base
        self.layer = layer
        self.owner = owner
        self.use_sliding = getattr(base, "use_sliding", False)
        self.is_linear = getattr(base, "is_linear", False)

    def __call__(self, x, mask=None, cache=None):
        result = self.base(x, mask, cache=cache)
        shift = self.owner._residual_active.get(self.layer)
        if shift is not None:
            result = result + shift.astype(result.dtype)
        tail = self.owner._tail_active.get(self.layer)
        if tail is not None:
            start_pos, vector = tail
            if result.shape[1] == 1:
                result = result + vector.astype(result.dtype)
            elif start_pos < result.shape[1]:
                result = result.at[:, start_pos:, :].add(vector.astype(result.dtype))
        donor = self.owner._residual_patches.get(self.layer)
        if donor is not None:
            pending = self.owner._generation_patch_pending
            # stream_generate sends all but the final prompt token in prefill
            # chunks, then processes that final token as the first singleton.
            # Patch that singleton exactly once, never generated tokens.
            if pending is not None:
                if result.shape[1] == 1 and self.layer in pending:
                    result = result.at[:, -1, :].add(donor.astype(result.dtype) - result[:, -1, :])
                    pending.remove(self.layer)
                    self.owner._patch_applied_layers.add(self.layer)
            elif result.shape[1] > 1:
                result = result.at[:, -1, :].add(donor.astype(result.dtype) - result[:, -1, :])
        for positions, vector in self.owner._token_residual_edits.get(self.layer, ()):
            if result.shape[1] > max(positions):
                for p in positions:
                    result = result.at[:, p, :].add(vector.astype(result.dtype))
        for positions, values in self.owner._token_residual_patches.get(self.layer, ()):
            if result.shape[1] > max(positions):
                for j, p in enumerate(positions):
                    result = result.at[:, p, :].add(values[j].astype(result.dtype) - result[:, p, :])
        return result


class AttentionHook(nn.Module):
    def __init__(self, base, layer: int, owner: "TheatreModel"):
        super().__init__()
        self.base = base
        self.layer = layer
        self.owner = owner

    def __call__(self, *args, **kwargs):
        projected = self.base(*args, **kwargs)
        if self.owner.capture_attentions:
            self.owner.all_attentions[self.layer] = projected
        shift = self.owner._attention_active.get(self.layer)
        if shift is not None:
            projected = projected + shift.astype(projected.dtype)
        return projected


class MLPHook(nn.Module):
    def __init__(self, base, layer: int, owner: "TheatreModel"):
        super().__init__()
        self.base = base
        self.layer = layer
        self.owner = owner

    def __call__(self, x):
        # Both supported dense MLPs use this post-SwiGLU input to down_proj.
        hidden = nn.silu(self.base.gate_proj(x)) * self.base.up_proj(x)
        if self.owner.capture_layer == self.layer:
            self.owner.captured_before = hidden
        if self.owner.capture_all:
            self.owner.all_hidden[self.layer] = hidden
        sparse = self.owner._active.get(self.layer)
        if sparse is not None:
            indices, values = sparse
            delta = mx.zeros_like(hidden).at[..., indices].add(values)
            hidden = hidden + delta
        for neuron, positions, scale in self.owner._token_neuron_edits.get(self.layer, ()):
            if hidden.shape[1] > max(positions):
                for p in positions:
                    before = hidden[:, p, neuron]
                    hidden = hidden.at[:, p, neuron].add((scale - 1) * before)
        for neuron, positions, delta in self.owner._token_neuron_offsets.get(self.layer, ()):
            if hidden.shape[1] > max(positions):
                for p in positions:
                    hidden = hidden.at[:, p, neuron].add(delta)
        if self.owner.capture_layer == self.layer:
            self.owner.captured_after = hidden
        projected = self.base.down_proj(hidden)
        if self.owner.capture_outputs:
            self.owner.all_outputs[self.layer] = projected
        shift = self.owner._output_active.get(self.layer)
        if shift is not None:
            projected = projected + shift.astype(projected.dtype)
        return projected


class ResidualCapture(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.use_sliding = getattr(base, "use_sliding", False)
        self.is_linear = getattr(base, "is_linear", False)
        self.output = None

    def __call__(self, x, mask=None, cache=None):
        self.output = self.base(x, mask, cache=cache)
        return self.output


class TheatreModel:
    """Serializes generations so one model cannot mix active persona states."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = getattr(model, "model_type", "")
        self.lock = RLock()
        self._active = {}
        self._output_active = {}
        self._attention_active = {}
        self._residual_active = {}
        self._tail_active = {}
        self._residual_patches = {}
        self._generation_patch_pending = None
        self._patch_applied_layers = set()
        self._token_residual_edits = {}
        self._token_residual_patches = {}
        self._token_neuron_edits = {}
        self._token_neuron_offsets = {}
        self.capture_layer = None
        self.captured_before = None
        self.captured_after = None
        self.capture_all = False
        self.capture_outputs = False
        self.capture_attentions = False
        self.all_hidden = {}
        self.all_outputs = {}
        self.all_attentions = {}
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or not layers:
            raise TypeError("Expected MLX model.model.layers")
        for i, layer in enumerate(layers):
            mlp = layer.mlp
            if not all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
                raise TypeError("Expected SwiGLU gate_proj, up_proj, and down_proj")
            layer.mlp = MLPHook(mlp, i, self)
            if hasattr(layer, "self_attn"):
                layer.self_attn = AttentionHook(layer.self_attn, i, self)
            if hasattr(layer, "linear_attn"):
                layer.linear_attn = AttentionHook(layer.linear_attn, i, self)
        self.layers = len(layers)
        self.width = int(layers[0].mlp.base.gate_proj.weight.shape[0])
        self.residual_width = int(layers[0].mlp.base.down_proj.weight.shape[0])

    @classmethod
    def from_cache(cls, name: str = DEFAULT_MODEL):
        model, tokenizer = load(cached_model_path(name))
        return cls(model, tokenizer)

    def _encode_interventions(self, interventions):
        grouped = {}
        for item in interventions:
            if not 0 <= item.layer < self.layers or not 0 <= item.neuron < self.width:
                raise ValueError(f"Invalid neuron coordinate: {item}")
            if not abs(item.delta) < float("inf"):
                raise ValueError("Intervention delta must be finite")
            if item.delta == 0:
                continue
            layer = grouped.setdefault(item.layer, {})
            layer[item.neuron] = layer.get(item.neuron, 0.0) + item.delta
        encoded = {}
        for layer, values in grouped.items():
            values = {neuron: value for neuron, value in values.items() if value != 0}
            if not values:
                continue
            indices = mx.array(list(values), dtype=mx.int32)
            deltas = mx.array(list(values.values()), dtype=mx.float32)
            mx.eval(indices, deltas)
            encoded[layer] = (indices, deltas)
        return encoded

    @contextmanager
    def using(self, interventions=(), output_shifts=(), attention_shifts=(),
              residual_shifts=(), tail_residual_shifts=(), residual_patches=(), token_residual_edits=(),
              token_residual_patches=(), token_neuron_edits=(), token_neuron_offsets=()):
        with self.lock:
            if (self._active or self._output_active or self._attention_active
                    or self._residual_active or self._tail_active or self._residual_patches
                    or self._token_residual_edits or self._token_residual_patches
                    or self._token_neuron_edits or self._token_neuron_offsets):
                raise RuntimeError("Nested or concurrent intervention state is not allowed")
            active = self._encode_interventions(interventions)
            def encode_shifts(shifts):
                encoded = {}
                for item in shifts:
                    if not 0 <= item.layer < self.layers:
                        raise ValueError(f"Invalid layer: {item.layer}")
                    vector = np.asarray(item.vector, dtype=np.float32)
                    if vector.shape != (self.residual_width,) or not np.isfinite(vector).all():
                        raise ValueError("Shift must be a finite residual-width vector")
                    encoded[item.layer] = encoded.get(item.layer, 0) + mx.array(vector)
                return encoded
            encoded_shifts = encode_shifts(output_shifts)
            encoded_attentions = encode_shifts(attention_shifts)
            encoded_residuals = encode_shifts(residual_shifts)
            encoded_tails = {}
            for item in tail_residual_shifts:
                if not 0 <= item.layer < self.layers or item.layer in encoded_tails:
                    raise ValueError("Invalid or duplicate tail residual shift layer")
                vector = np.asarray(item.vector, dtype=np.float32)
                if vector.shape != (self.residual_width,) or not np.isfinite(vector).all():
                    raise ValueError("Invalid tail residual shift vector")
                if item.start_pos < 0:
                    raise ValueError("Tail residual start position must be nonnegative")
                encoded_tails[item.layer] = (item.start_pos, mx.array(vector))
            encoded_patches = {}
            for item in residual_patches:
                if not 0 <= item.layer < self.layers or item.layer in encoded_patches:
                    raise ValueError(f"Invalid or duplicate residual patch layer: {item.layer}")
                donor = np.asarray(item.donor, dtype=np.float32)
                if donor.shape != (self.residual_width,) or not np.isfinite(donor).all():
                    raise ValueError("Residual patch must be a finite residual-width vector")
                encoded_patches[item.layer] = mx.array(donor)
            if set(encoded_patches) & set(encoded_residuals):
                raise ValueError("Cannot shift and patch the same residual layer")
            def positions_of(item):
                positions = tuple(int(p) for p in item.positions)
                if not positions or min(positions) < 0 or len(set(positions)) != len(positions):
                    raise ValueError("Token positions must be unique nonnegative integers")
                return positions
            token_edits, token_patches, neuron_edits, neuron_offsets = {}, {}, {}, {}
            for item in token_residual_edits:
                if not 0 <= item.layer < self.layers:
                    raise ValueError("Invalid token residual edit layer")
                vector = np.asarray(item.vector, dtype=np.float32)
                if vector.shape != (self.residual_width,) or not np.isfinite(vector).all():
                    raise ValueError("Invalid token residual edit vector")
                token_edits.setdefault(item.layer, []).append((positions_of(item), mx.array(vector)))
            for item in token_residual_patches:
                if not 0 <= item.layer < self.layers:
                    raise ValueError("Invalid token residual patch layer")
                positions = positions_of(item)
                donor = np.asarray(item.donor, dtype=np.float32)
                if donor.shape != (len(positions), self.residual_width) or not np.isfinite(donor).all():
                    raise ValueError("Invalid token residual donor array")
                token_patches.setdefault(item.layer, []).append((positions, mx.array(donor)))
            for item in token_neuron_edits:
                if not 0 <= item.layer < self.layers or not 0 <= item.neuron < self.width:
                    raise ValueError("Invalid token neuron coordinate")
                if not np.isfinite(item.scale):
                    raise ValueError("Invalid token neuron scale")
                neuron_edits.setdefault(item.layer, []).append((item.neuron, positions_of(item), item.scale))
            for item in token_neuron_offsets:
                if not 0 <= item.layer < self.layers or not 0 <= item.neuron < self.width:
                    raise ValueError("Invalid token neuron coordinate")
                if not np.isfinite(item.delta):
                    raise ValueError("Invalid token neuron offset")
                neuron_offsets.setdefault(item.layer, []).append((item.neuron, positions_of(item), item.delta))
            self._active = active
            self._output_active = encoded_shifts
            self._attention_active = encoded_attentions
            self._residual_active = encoded_residuals
            self._tail_active = encoded_tails
            self._residual_patches = encoded_patches
            self._token_residual_edits = token_edits
            self._token_residual_patches = token_patches
            self._token_neuron_edits = neuron_edits
            self._token_neuron_offsets = neuron_offsets
            originals = {}
            try:
                for layer in set(encoded_residuals) | set(encoded_tails) | set(encoded_patches) | set(token_edits) | set(token_patches):
                    block = self.model.model.layers[layer]
                    originals[layer] = block
                    self.model.model.layers[layer] = ResidualEditHook(block, layer, self)
                yield self
            finally:
                for layer, block in originals.items():
                    self.model.model.layers[layer] = block
                self._active = {}
                self._output_active = {}
                self._attention_active = {}
                self._residual_active = {}
                self._tail_active = {}
                self._residual_patches = {}
                self._generation_patch_pending = None
                self._token_residual_edits = {}
                self._token_residual_patches = {}
                self._token_neuron_edits = {}
                self._token_neuron_offsets = {}
                self.capture_layer = None
                self.captured_before = None
                self.captured_after = None
                self.capture_all = False
                self.capture_outputs = False
                self.capture_attentions = False
                self.all_hidden = {}
                self.all_outputs = {}
                self.all_attentions = {}

    def prompt(self, messages):
        options = {"enable_thinking": False} if self.model_type == "qwen3_5" else {}
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, **options)

    def generate(self, messages, interventions=(), max_tokens=96, temperature=0.0,
                 seed=0, output_shifts=(), attention_shifts=(),
                 residual_shifts=(), tail_residual_shifts=(), residual_patches=(), token_residual_edits=(),
                 token_residual_patches=(), token_neuron_edits=(), token_neuron_offsets=()):
        with self.using(interventions, output_shifts, attention_shifts,
                        residual_shifts, tail_residual_shifts, residual_patches, token_residual_edits,
                        token_residual_patches, token_neuron_edits, token_neuron_offsets):
            self._generation_patch_pending = set(self._residual_patches)
            self._patch_applied_layers = set()
            mx.random.seed(seed)
            started = perf_counter()
            last = None
            parts = []
            for last in stream_generate(self.model, self.tokenizer, self.prompt(messages),
                                        max_tokens=max_tokens,
                                        sampler=make_sampler(temp=temperature,
                                                             top_p=0.9 if temperature else 0.0)):
                parts.append(last.text)
            return {"text": "".join(parts).strip(),
                    "seconds": perf_counter() - started,
                    "prompt_tokens": last.prompt_tokens if last else 0,
                    "generated_tokens": last.generation_tokens if last else 0,
                    "prompt_tps": last.prompt_tps if last else 0.0,
                    "generation_tps": last.generation_tps if last else 0.0,
                    "peak_gb": last.peak_memory if last else 0.0,
                    "patch_applied_layers": sorted(self._patch_applied_layers)}

    def activation_pair(self, messages, completion: str, layer: int, interventions=(),
                        max_answer_tokens=32):
        """Return post-SwiGLU arrays for a teacher-forced completion."""
        prefix = self.tokenizer.encode(self.prompt(messages), add_special_tokens=False)
        answer = self.tokenizer.encode(completion, add_special_tokens=False)
        with self.using(interventions):
            self.capture_layer = layer
            self.model(mx.array([prefix + answer]))
            start = len(prefix)
            end = start + min(len(answer), max_answer_tokens)
            before = self.captured_before[0, start:end].astype(mx.float32)
            after = self.captured_after[0, start:end].astype(mx.float32)
            mx.eval(before, after)
            return before, after

    def activation_means(self, messages, completion: str, token_limit: int):
        """Capture per-layer mean post-SwiGLU activity for matched answer tokens."""
        prefix = self.tokenizer.encode(self.prompt(messages), add_special_tokens=False)
        answer = self.tokenizer.encode(completion, add_special_tokens=False)
        if not answer or token_limit < 1:
            raise ValueError("Need answer tokens to capture")
        with self.using():
            self.capture_all = True
            self.model(mx.array([prefix + answer]))
            start = len(prefix)
            end = start + min(len(answer), token_limit)
            means = [mx.mean(self.all_hidden[i][0, start:end].astype(mx.float32), axis=0)
                     for i in range(self.layers)]
            mx.eval(*means)
            return np.stack([np.asarray(x) for x in means])

    def activation_means_with_output(self, messages, completion: str, token_limit: int):
        """Mean post-SwiGLU activations and unshifted MLP outputs at every layer."""
        prefix = self.tokenizer.encode(self.prompt(messages), add_special_tokens=False)
        answer = self.tokenizer.encode(completion, add_special_tokens=False)
        if not answer or token_limit < 1:
            raise ValueError("Need answer tokens to capture")
        with self.using():
            self.capture_all = True
            self.capture_outputs = True
            self.model(mx.array([prefix + answer]))
            start = len(prefix)
            end = start + min(len(answer), token_limit)
            hidden = [mx.mean(self.all_hidden[i][0, start:end].astype(mx.float32), axis=0)
                      for i in range(self.layers)]
            outputs = [mx.mean(self.all_outputs[i][0, start:end].astype(mx.float32), axis=0)
                       for i in range(self.layers)]
            mx.eval(*hidden, *outputs)
            return (np.stack([np.asarray(x) for x in hidden]),
                    np.stack([np.asarray(x) for x in outputs]))

    def activation_means_with_components(self, messages, completion: str, token_limit: int):
        """Mean MLP hidden, MLP output, and attention output over answer tokens."""
        prefix = self.tokenizer.encode(self.prompt(messages), add_special_tokens=False)
        answer = self.tokenizer.encode(completion, add_special_tokens=False)
        if not answer or token_limit < 1:
            raise ValueError("Need answer tokens to capture")
        with self.using():
            self.capture_all = self.capture_outputs = self.capture_attentions = True
            self.model(mx.array([prefix + answer]))
            start = len(prefix)
            end = start + min(len(answer), token_limit)
            groups = (self.all_hidden, self.all_outputs, self.all_attentions)
            means = [[mx.mean(group[i][0, start:end].astype(mx.float32), axis=0)
                      for i in range(self.layers)] for group in groups]
            mx.eval(*(value for group in means for value in group))
            return tuple(np.stack([np.asarray(value) for value in group]) for group in means)

    def projected_delta_unit(self, layer: int, neuron: int, messages):
        """Activation offset whose down-projected norm equals one residual norm."""
        with self.lock:
            if (self._active or self._output_active or self._attention_active
                    or self._residual_active or self._residual_patches):
                raise RuntimeError("Cannot calibrate during an active intervention")
            block = self.model.model.layers[layer]
            down = block.mlp.base.down_proj
            one = mx.zeros((1, 1, self.width), dtype=mx.float32).at[..., neuron].add(1)
            zero = mx.zeros_like(one)
            column = float(mx.linalg.norm((down(one) - down(zero)).astype(mx.float32)))
            capture = ResidualCapture(block)
            self.model.model.layers[layer] = capture
            try:
                ids = self.tokenizer.encode(self.prompt(messages), add_special_tokens=False)
                self.model(mx.array([ids]))
                residual = float(mx.median(mx.linalg.norm(
                    capture.output[0, 1:].astype(mx.float32), axis=-1)))
            finally:
                self.model.model.layers[layer] = block
            if column == 0:
                raise ValueError("Neuron has zero projected norm")
            return {"residual_norm": residual, "column_norm": column,
                    "delta_per_residual": residual / column}
