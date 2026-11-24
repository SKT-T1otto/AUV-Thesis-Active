"""Prepare the outcome-free Chapter 3 launch assignment registry, without a mission rollout."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from chapter3_bser.learning.config import scenario_fingerprint
from core.scenarios.ch3_generator_impl import build_scenario_manifests
from project_paths import ROOT


CONFIG = ROOT / "configs/launch_assignment_v1/study.json"
LOCK = ROOT / "provenance/launch_assignment_v1.json"
SPLITS = ("development", "calibration", "blind")
FAMILY_KEYS = ("physical", "launch_positions", "layout", "trajectory", "motion_origin", "scenario_seed")


def read(path):
    return json.loads(path.read_bytes())


def digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_vector(value, shape):
    data = np.array(value, dtype=np.float32)
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError("invalid physical initial conditions")
    data[data == 0] = 0  # +0 and -0 have the same physical interpretation.
    return data.tolist()


def scene_identity(scene):
    obstacles = scene["obstacles"]
    if digest(obstacles) != scene["obstacle_layout_sha256"]:
        raise ValueError("layout checksum disagrees with native obstacle data")
    layout = sorted([[physical_vector(o["center"], (3,)), physical_vector(o["size"], (3,))]
                     for o in obstacles])
    trajectory = scene["target_trajectory_sha256"]
    if len(trajectory) != 64 or any(c not in "0123456789abcdef" for c in trajectory):
        raise ValueError("invalid native trajectory checksum")
    return {
        "scenario_id": scene["scenario_id"],
        "scenario_seed": scene["scenario_seed"],
        "physical": scenario_fingerprint(scene),
        "launch_positions": digest(physical_vector(scene["initial_agent_positions"], (4, 3))),
        "layout": digest(layout),
        "trajectory": trajectory,
        # The native trajectory hash also includes the layout. Sharing a motion
        # origin conservatively keeps even cross-layout trajectory copies together.
        "motion_origin": digest(physical_vector(scene["target_initial_position"], (3,))),
    }


def verify_history(history, lock):
    """Use original archive bytes only for an explicitly requested local audit."""
    for source in history["sources"]:
        path = ROOT / source["path"]
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"historical source changed: {path}")
        scenes = read(path)[source["field"]]
        if [scene_identity(scene) for scene in scenes] != source["scenes"]:
            raise ValueError(f"historical projection changed: {path}")
    for group in ("historical_manifest_sha256", "historical_regression_file_sha256"):
        for relative, expected in lock[group].items():
            if file_hash(ROOT / relative) != expected:
                raise ValueError(f"historical reference changed: {relative}")


def partition(rows, fresh_counts, split_seed):
    """Assign connected components. A history alias pins the entire component to development."""
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for i, row in enumerate(rows):
        for key in FAMILY_KEYS:
            first = seen.setdefault((key, row[key]), i)
            parents[root(i)] = root(first)

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)

    families = []
    for members in groups.values():
        family_id = digest(sorted({r["physical"] for r in members}))
        families.append({
            "family_id": family_id,
            "split": "development" if any(r["source"] != "fresh" for r in members) else None,
            "members": sorted(members, key=lambda r: (r["source"], r["index"])),
        })
    fresh = sorted((f for f in families if f["split"] is None),
                   key=lambda f: digest([split_seed, f["family_id"]]))
    if len(fresh) != sum(fresh_counts.values()):
        raise ValueError("independent fresh-family quota not met; no replacement or silent resampling")
    start = 0
    for split in SPLITS:
        end = start + fresh_counts[split]
        for family in fresh[start:end]:
            family["split"] = split
        start = end
    return sorted(families, key=lambda f: f["family_id"])


def stream_seed(streams, family_id, stream, boat, repeat=0):
    offers = streams["candidate_streams"]
    if stream not in [*offers, streams["continuation_stream"]]:
        raise ValueError("unknown study random stream")
    if type(boat) is not int or boat not in range(3) or type(repeat) is not int or repeat < 0:
        raise ValueError("invalid search boat or replicate for this stream")
    if stream in offers and repeat != 0:
        raise ValueError("a paired continuation cannot redraw the departure offers")
    return int(digest([streams["master_seed"], family_id, stream, boat, repeat])[:16], 16)


def prepare(*, audit_history=False):
    lock = read(LOCK)
    for relative, expected in lock["prepare_file_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"prepare source or input changed: {relative}")
    study = read(CONFIG)
    native = study["native_inputs"]
    history = read(ROOT / study["historical_registry"])
    if audit_history:
        verify_history(history, lock)
        proposer = study["candidates"]["proposer_reference"]
        if file_hash(ROOT / proposer["path"]) != proposer["file_sha256"]:
            raise ValueError("historical proposer artifact changed")
    rows = [dict(row, source=source["name"], index=i) for source in history["sources"]
            for i, row in enumerate(source["scenes"])]
    exclusions = read(ROOT / "configs/r123_expansion_v1/exclusions.json")["fingerprints"]
    if not set(exclusions).issubset({r["physical"] for r in rows}):
        raise ValueError("previously screened physical scenes missing from development history")
    counts = native["fresh_families"]
    if set(counts) != set(SPLITS) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("expected a positive family quota for each study split")
    batch = build_scenario_manifests(sum(counts.values()), native["generator_seed"],
                                     split=native["native_split"], profiles=native["profile"])[native["profile"]]
    identities = [scene_identity(scene) for scene in batch["scenarios"]]
    if digest(identities) != lock["fresh_identity_sha256"]:
        raise ValueError("native generated pool differs from the frozen physical identities")
    fresh = [dict(identity, source="fresh", index=i) for i, identity in enumerate(identities)]
    families = partition(rows + fresh, counts, native["family_split_seed"])
    cells = []
    if study["holdout"]["search"] not in study["continuations"] or study["holdout"]["terminal"] not in study["terminal_modes"]:
        raise ValueError("held-out search or terminal mode is missing from the study")
    for search in study["continuations"]:
        for terminal in study["terminal_modes"]:
            held = (search, terminal) == (study["holdout"]["search"], study["holdout"]["terminal"])
            cells.append(dict(search=search, terminal=terminal, access={
                "development": "sealed" if held else "fit",
                "calibration": "sealed" if held else "calibrate_only",
                "blind": "sealed",
            }))
    return {
        "schema": "auv.launch_assignment.preparation.v1",
        "source": {"base_commit": lock["base_commit"], "source_manifest_sha256": file_hash(LOCK)},
        "study": study, "cells": cells, "family_counts": dict(Counter(f["split"] for f in families)),
        "families": families,
    }


def publish(directory, manifest):
    """Publish one complete file; identical repeats are harmless even across processes."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "manifest.json"
    data = (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(dir=directory)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as staged:
            staged.write(data)
        try:
            if os.name == "nt":
                os.rename(temp_path, target)  # Windows rename is atomic and refuses an existing name.
            else:
                os.link(temp_path, target)  # POSIX rename would replace the existing manifest.
        except FileExistsError:
            if target.read_bytes() != data:
                raise FileExistsError(f"refusing to replace a different manifest: {target}") from None
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="freeze family assignments and the held-out configuration")
    prep.add_argument("--output", type=Path, required=True, help="directory for manifest.json")
    prep.add_argument("--verify-history", action="store_true", help="also audit original local archives and parent locks")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = prepare(audit_history=args.verify_history)
    path = publish(args.output, manifest)
    print(json.dumps({"manifest": str(path), "sha256": file_hash(path), "families": manifest["family_counts"]}))


if __name__ == "__main__":
    main()
