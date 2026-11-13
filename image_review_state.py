"""Atomic, versioned review checkpoints and bounded undo/redo without pickle."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from image_review import Candidate, ClickPrompt


class CandidateData(TypedDict):
    mask: str | None
    score: float
    prompt: str
    identity: str
    visibility: str


@dataclass
class ReviewState:
    candidates: list[Candidate]
    instances: list[dict[str, str]]
    shape: tuple[int, int] = (0, 0)
    assignments: dict[str, list[int]] = field(default_factory=dict)
    states: dict[str, str] = field(default_factory=dict)
    excluded: set[int] = field(default_factory=set)
    ignored: set[int] = field(default_factory=set)
    before_edits: dict[str, Candidate] = field(default_factory=dict)
    refinements: dict[str, Candidate] = field(default_factory=dict)
    cleaned: dict[str, Candidate] = field(default_factory=dict)
    active_prompts: dict[str, ClickPrompt] = field(default_factory=dict)
    click_prompts: dict[int, ClickPrompt] = field(default_factory=dict)
    generated: set[int] = field(default_factory=set)
    removed: set[int] = field(default_factory=set)
    ordered: dict[str, Candidate] = field(default_factory=dict)
    depth_order: list[str] = field(default_factory=list)
    ownership: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    locked: set[str] = field(default_factory=set)


def state_path(review_log: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return review_log.parent / f"{review_log.stem}_state" / f"{digest}.npz"


def encode_state(state: ReviewState) -> bytes:
    arrays: dict[str, NDArray[np.uint8] | NDArray[np.str_]] = {}

    def mask(value: NDArray[np.bool_] | None) -> str | None:
        if value is None:
            return None
        name = f"mask_{len(arrays)}"
        arrays[name] = np.packbits(value)
        return name

    def candidate(value: Candidate) -> CandidateData:
        return {
            "mask": mask(value.mask),
            "score": value.score,
            "prompt": value.prompt,
            "identity": value.identity,
            "visibility": value.visibility,
        }

    # Active and indexed histories must refer to the same prompt after restore.
    prompts = list(state.click_prompts.values()) + list(state.active_prompts.values())
    unique_prompts = list({id(prompt): prompt for prompt in prompts}.values())
    prompt_indices = {id(prompt): index for index, prompt in enumerate(unique_prompts)}
    metadata = {
        "shape": state.shape,
        "candidates": [candidate(value) for value in state.candidates],
        "instances": state.instances,
        "assignments": state.assignments,
        "states": state.states,
        "excluded": sorted(state.excluded),
        "ignored": sorted(state.ignored),
        "removed": sorted(state.removed),
        "generated": sorted(state.generated),
        "locked": sorted(state.locked),
        "before_edits": {
            key: candidate(value) for key, value in state.before_edits.items()
        },
        "refinements": {
            key: candidate(value) for key, value in state.refinements.items()
        },
        "cleaned": {key: candidate(value) for key, value in state.cleaned.items()},
        "ordered": {key: candidate(value) for key, value in state.ordered.items()},
        "ownership": {key: mask(value) for key, value in state.ownership.items()},
        "depth_order": state.depth_order,
        "prompts": [
            {
                "instance_id": value.instance_id,
                "points": value.points,
                "seed": mask(value.seed),
                "candidate_index": value.candidate_index,
                "separate": value.separate,
            }
            for value in unique_prompts
        ],
        "active_prompts": {
            key: prompt_indices[id(value)]
            for key, value in state.active_prompts.items()
        },
        "click_prompts": {
            str(key): prompt_indices[id(value)]
            for key, value in state.click_prompts.items()
        },
    }
    arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    with io.BytesIO() as output:
        np.savez_compressed(output, **arrays)
        return output.getvalue()


def decode_state(data: bytes) -> ReviewState:
    from image_review import Candidate, ClickPrompt

    with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"]))
        shape = metadata["shape"]

        def mask(name: str | None) -> NDArray[np.bool_] | None:
            if name is None:
                return None
            if shape is None or len(shape) != 2:
                raise ValueError("Checkpoint mask dimensions are missing.")
            pixels = np.unpackbits(arrays[name], count=int(np.prod(shape)))
            return pixels.reshape(shape).astype(np.bool_)

        def candidate(value: CandidateData) -> Candidate:
            pixels = mask(value["mask"])
            if pixels is None:
                raise ValueError("Checkpoint candidate has no mask.")
            return Candidate(
                pixels,
                float(value["score"]),
                value["prompt"],
                value["identity"],
                value["visibility"],
            )

        prompts = [
            ClickPrompt(
                value["instance_id"],
                [(float(x), float(y), int(label)) for x, y, label in value["points"]],
                mask(value["seed"]),
                value["candidate_index"],
                value["separate"],
            )
            for value in metadata["prompts"]
        ]
        ownership: dict[str, NDArray[np.bool_]] = {}
        for key, value in metadata["ownership"].items():
            pixels = mask(value)
            if pixels is None:
                raise ValueError("Checkpoint ownership has no mask.")
            ownership[key] = pixels
        return ReviewState(
            candidates=[candidate(value) for value in metadata["candidates"]],
            instances=metadata["instances"],
            shape=(int(shape[0]), int(shape[1])),
            assignments=metadata["assignments"],
            states=metadata["states"],
            excluded=set(metadata["excluded"]),
            ignored=set(metadata["ignored"]),
            removed=set(metadata["removed"]),
            generated=set(metadata["generated"]),
            locked=set(metadata["locked"]),
            before_edits={
                key: candidate(value)
                for key, value in metadata.get("before_edits", {}).items()
            },
            refinements={
                key: candidate(value)
                for key, value in metadata["refinements"].items()
            },
            cleaned={
                key: candidate(value) for key, value in metadata["cleaned"].items()
            },
            ordered={
                key: candidate(value) for key, value in metadata["ordered"].items()
            },
            ownership={
                key: value for key, value in ownership.items() if value is not None
            },
            depth_order=metadata["depth_order"],
            active_prompts={
                key: prompts[value]
                for key, value in metadata["active_prompts"].items()
            },
            click_prompts={
                int(key): prompts[value]
                for key, value in metadata["click_prompts"].items()
            },
        )


class ReviewHistory:
    def __init__(self, path: Path | None = None, fingerprint: str = "") -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.frames: list[bytes] = []
        self.position = -1
        self.baseline = b""
        if path is not None and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as data:
                    if int(data["version"]) != 1:
                        raise ValueError("unsupported checkpoint version")
                    if str(data["fingerprint"]) != fingerprint:
                        print("Review checkpoint is stale; starting fresh predictions.")
                        return
                    self.baseline = data["baseline"].tobytes()
                    self.frames = [
                        data[f"frame_{index}"].tobytes()
                        for index in range(int(data["count"]))
                    ]
                    self.position = int(data["position"])
                    if not 0 <= self.position < len(self.frames):
                        raise ValueError("invalid history position")
                    decode_state(self.frames[self.position])
                    decode_state(self.baseline)
            except (OSError, ValueError, KeyError, TypeError, BadZipFile) as error:
                raise ValueError(
                    f"Cannot restore review checkpoint {path}: {error}"
                ) from error

    @property
    def current(self) -> ReviewState | None:
        return decode_state(self.frames[self.position]) if self.frames else None

    def commit(self, state: ReviewState) -> None:
        data = encode_state(state)
        if self.frames and data == self.frames[self.position]:
            return
        if not self.baseline:
            self.baseline = data
        self.frames = self.frames[: self.position + 1]
        self.frames.append(data)
        self.frames = self.frames[-40:]
        self.position = len(self.frames) - 1
        self.save()

    def move(self, offset: int) -> ReviewState:
        position = self.position + offset
        if not 0 <= position < len(self.frames):
            raise ValueError("Nothing to undo." if offset < 0 else "Nothing to redo.")
        state = decode_state(self.frames[position])
        self.position = position
        self.save()
        return state

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=self.path.parent, suffix=".npz")
        try:
            with os.fdopen(descriptor, "wb") as output:
                np.savez_compressed(
                    output,
                    version=np.asarray(1),
                    fingerprint=np.asarray(self.fingerprint),
                    baseline=np.frombuffer(self.baseline, dtype=np.uint8),
                    count=np.asarray(len(self.frames)),
                    position=np.asarray(self.position),
                    **{
                        f"frame_{index}": np.frombuffer(frame, dtype=np.uint8)
                        for index, frame in enumerate(self.frames)
                    },
                )
            Path(name).replace(self.path)
        finally:
            Path(name).unlink(missing_ok=True)
