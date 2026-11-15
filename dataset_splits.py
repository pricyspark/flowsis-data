"""Physical-instance holdouts followed by capture-group holdouts."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Literal

from dataset_schema import ImageRecord


def capture_groups(
    rows: list[ImageRecord], group_by: Literal["sample", "session"]
) -> list[list[int]]:
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        if not row["sample_id"] and not row["video_id"]:
            raise ValueError(f"Missing Sample ID for {row['image_id']}")
        keys = [("sample", row["sample_id"]), ("video", row["video_id"])]
        if group_by == "session":
            if not row["capture_session_id"]:
                raise ValueError(f"Missing capture session for {row['image_id']}")
            keys.append(("session", row["capture_session_id"]))
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return sorted(
        groups.values(), key=lambda group: min(rows[i]["image_id"] for i in group)
    )


def split_tool_samples(
    rows: list[ImageRecord],
    *,
    unseen_tool_size: float = 0.1,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
    group_by: Literal["sample", "session"] = "sample",
    unseen_tool_ids: list[str] | None = None,
) -> tuple[list[str], dict]:
    """Quarantine unseen tools, then use GroupShuffleSplit on whole captures.

    Validation/test fractions apply to remaining capture groups, not frames.
    Training anchors preserve every remaining tool and its supervised classes.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if group_by not in {"sample", "session"}:
        raise ValueError("group_by must be sample or session")
    if not rows:
        raise ValueError("No rows to split")
    if not math.isfinite(unseen_tool_size) or not 0 < unseen_tool_size < 1:
        raise ValueError("unseen_tool_size must be strictly between 0 and 1")
    if (
        any(not math.isfinite(x) or x < 0 for x in (validation_size, test_size))
        or validation_size + test_size >= 1
    ):
        raise ValueError(
            "Validation/test fractions must be nonnegative and sum to less than 1"
        )
    groups = capture_groups(rows, group_by)
    group_tools = [
        {tool for i in group for tool in rows[i]["tool_ids"]} for group in groups
    ]
    tools = set().union(*group_tools)
    if not tools:
        raise ValueError("Physical Tool IDs are required")
    supervised: list[set[str]] = []
    tool_labels: dict[str, set[int]] = defaultdict(set)
    group_labels: list[set[int]] = []
    for group in groups:
        visible: set[str] = set()
        labels: set[int] = set()
        for i in group:
            row = rows[i]
            if not row["base_eligible"]:
                continue
            for obj in row["objects"]:
                if obj["tool_id"] not in row["tool_ids"]:
                    raise ValueError(
                        f"Object Tool ID absent from row: {row['image_id']}"
                    )
                if obj["class_valid"] and obj["mask_valid"] and obj["visible"]:
                    visible.add(obj["tool_id"])
                    tool_labels[obj["tool_id"]].add(obj["category"])
                    labels.add(obj["category"])
        supervised.append(visible)
        group_labels.append(labels)
    if tools - tool_labels.keys():
        raise ValueError(
            f"Tools without visible base supervision: {sorted(tools - tool_labels.keys())}"
        )

    def feasible(held: set[str]) -> bool:
        available = [i for i, ids in enumerate(group_tools) if not ids & held]
        remaining = set().union(*(supervised[i] for i in available))
        labels = set().union(*(group_labels[i] for i in available))
        return remaining == tools - held and labels == set().union(*group_labels)

    candidates = sorted(tools)
    random.Random(seed).shuffle(candidates)
    held: set[str] = set()
    if unseen_tool_ids is not None:
        held = set(unseen_tool_ids)
        if not held or not held <= tools:
            raise ValueError(
                "Explicit unseen Tool IDs must be a nonempty subset of eligible tools"
            )
        if not feasible(held):
            raise ValueError(
                "Explicit unseen tools remove training class/tool coverage after capture quarantine"
            )
    else:
        requested = math.ceil(len(tools) * unseen_tool_size)
        for tool in candidates:
            if feasible(held | {tool}):
                held.add(tool)
            if len(held) == requested:
                break
        if len(held) < requested:
            raise ValueError(
                f"Only {len(held)} of {requested} requested unseen tools could be held out "
                "without losing supervised classes/known tools. Reduce unseen-tool-size, "
                "provide explicit unseen-tool-ids, or collect independent captures."
            )
    quarantined = {i for i, ids in enumerate(group_tools) if ids & held}
    available = [i for i in range(len(groups)) if i not in quarantined]
    # Reserve few captures covering all known physical tools before random holdout.
    uncovered = tools - held
    uncovered_labels = set().union(*group_labels)
    anchors: set[int] = set()
    ranks = list(available)
    random.Random(seed).shuffle(ranks)
    rank = {value: i for i, value in enumerate(ranks)}
    while uncovered or uncovered_labels:
        selected = min(
            available,
            key=lambda i: (
                -len(supervised[i] & uncovered)
                - len(group_labels[i] & uncovered_labels),
                len(groups[i]),
                rank[i],
            ),
        )
        anchors.add(selected)
        uncovered -= supervised[selected]
        uncovered_labels -= group_labels[selected]
    val_count = math.ceil(len(available) * validation_size)
    test_count = math.ceil(len(available) * test_size)
    pool = [i for i in available if i not in anchors]
    if val_count + test_count > len(pool):
        raise ValueError(
            f"Need {val_count + test_count} held-out capture groups but only {len(pool)} "
            "remain after training coverage. Reduce validation/test sizes or add captures."
        )

    def draw(
        pool: list[int], count: int, draw_seed: int
    ) -> tuple[list[int], list[int]]:
        if not count:
            return pool, []
        if count == len(pool):
            return [], pool
        keep, holdout = next(
            GroupShuffleSplit(
                n_splits=1, test_size=count, random_state=draw_seed
            ).split(pool, groups=pool)
        )
        return [pool[int(i)] for i in keep], [pool[int(i)] for i in holdout]

    # Blind metadata constraints prevent a nominal temporal validation split with
    # no video. Enforce each condition only when enough unanchored groups exist.
    feature_groups = {
        "video": {i for i in available if any(rows[j]["video_id"] for j in groups[i])},
        "multi_tool": {
            i for i in available if any(len(rows[j]["objects"]) > 1 for j in groups[i])
        },
        "background": {
            i for i in available if any(not rows[j]["objects"] for j in groups[i])
        },
    }
    holdouts = int(val_count > 0) + int(test_count > 0)
    enforced = {
        feature: ids
        for feature, ids in feature_groups.items()
        if len(ids) >= 1 + holdouts
        and len(ids - anchors) >= holdouts + int(not bool(ids & anchors))
    }
    best_draw: tuple[list[int], list[int], int] | None = None
    best_error = math.inf
    remaining_rows = sum(len(groups[i]) for i in available)
    for attempt in range(1024):
        remaining, test_groups = draw(pool, test_count, seed + attempt * 2)
        remaining, val_groups = draw(remaining, val_count, seed + attempt * 2 + 1)
        cohorts = [anchors | set(remaining)]
        if val_count:
            cohorts.append(set(val_groups))
        if test_count:
            cohorts.append(set(test_groups))
        if all(ids & cohort for ids in enforced.values() for cohort in cohorts):
            error = abs(
                sum(len(groups[i]) for i in val_groups)
                - validation_size * remaining_rows
            )
            error += abs(
                sum(len(groups[i]) for i in test_groups) - test_size * remaining_rows
            )
            if error < best_error:
                best_error = error
                best_draw = val_groups, test_groups, attempt + 1
    if best_draw is None:
        raise ValueError(
            "Could not allocate capture groups with video/multi-tool/background coverage; add captures or adjust fractions/grouping"
        )
    val_groups, test_groups, selected_draw = best_draw
    group_splits = dict.fromkeys(available, "train")
    group_splits.update(dict.fromkeys(quarantined, "test_unseen"))
    group_splits.update(dict.fromkeys(test_groups, "test_known"))
    group_splits.update(dict.fromkeys(val_groups, "validation"))
    assignments = [""] * len(rows)
    for group_index, group in enumerate(groups):
        for i in group:
            assignments[i] = group_splits[group_index]
    per_split = {}
    for name in ("train", "validation", "test_known", "test_unseen"):
        ids = {
            tool
            for row, split in zip(rows, assignments, strict=True)
            if split == name
            for tool in row["tool_ids"]
        }
        per_split[name] = {
            "tool_ids": sorted(ids),
            "capture_groups": sum(s == name for s in group_splits.values()),
        }
    assert not set(per_split["train"]["tool_ids"]) & held
    assert set(per_split["test_known"]["tool_ids"]) <= set(
        per_split["train"]["tool_ids"]
    )
    return assignments, {
        "method": "tool quarantine + training coverage anchors + sklearn GroupShuffleSplit",
        "group_by": group_by,
        "fraction_units": "unseen_tool_size: physical IDs; validation/test_size: remaining capture groups",
        "unseen_tool_ids": sorted(held),
        "known_companion_tool_ids_in_unseen_test": sorted(
            set(per_split["test_unseen"]["tool_ids"]) - held
        ),
        "requested_unseen_tool_size": unseen_tool_size,
        "actual_unseen_tool_fraction": len(held) / len(tools),
        "enforced_capture_conditions": sorted(enforced),
        "coverage_warnings": [
            f"{name}: insufficient free groups for coverage in every known-tool split"
            for name in feature_groups
            if name not in enforced
        ],
        "capture_draw_candidates": 1024,
        "selected_capture_draw": selected_draw,
        "capture_selection_rule": "metadata coverage, then minimum validation/test row-count deviation; no model scores",
        "selection": (
            "explicit"
            if unseen_tool_ids is not None
            else "seeded coverage-constrained selection"
        ),
        "splits": per_split,
        "assignments": {
            row["image_id"]: split for row, split in zip(rows, assignments, strict=True)
        },
        "note": "test_unseen is a scene cohort; score held-out tool targets separately from known companions. Old checkpoints may already have seen these Tool IDs.",
    }
