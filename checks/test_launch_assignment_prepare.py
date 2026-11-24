import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import launch_assignment_study as study


def row(i, source="fresh", **changes):
    result = {key: f"{key}.{i}" for key in study.FAMILY_KEYS}
    result.update(source=source, index=i, scenario_id="reused_native_name", scenario_seed=i)
    return dict(result, **changes)


class FamilyTests(unittest.TestCase):
    def test_source_drift_stops_before_native_generation(self):
        with patch.object(study, "file_hash", return_value="different"), patch.object(study, "build_scenario_manifests") as gen:
            with self.assertRaisesRegex(ValueError, "source or input changed"):
                study.prepare()
            gen.assert_not_called()

    def test_shared_key_chains_and_aliases_are_pinned_to_history(self):
        old = row(1, "archive")
        bridge = row(2, layout=old["layout"])
        prefix = row(3, launch_positions=bridge["launch_positions"])
        trajectory = row(4, trajectory=prefix["trajectory"])
        cross_layout = row(5, motion_origin=trajectory["motion_origin"])
        rows = [old, bridge, prefix, trajectory, cross_layout, row(6), row(7), row(8)]
        counts = dict(development=1, calibration=1, blind=1)
        result = study.partition(rows, counts, 23)
        self.assertEqual(result, study.partition(list(reversed(rows)), counts, 23))
        joined = next(f for f in result if len(f["members"]) > 1)
        self.assertEqual(joined["split"], "development")
        self.assertEqual({r["index"] for r in joined["members"]}, {1, 2, 3, 4, 5})
        self.assertEqual(len(result), 4)  # A reused native name is not a family key.

    def test_fresh_quota_collision_is_reported_without_replacement(self):
        rows = [row(0, "archive"), row(1), row(2), row(3, scenario_seed=0)]
        with self.assertRaisesRegex(ValueError, "quota"):
            study.partition(rows, dict(development=1, calibration=1, blind=1), 11)

    def test_native_identity_uses_geometry_independently_of_names_and_box_order(self):
        original = study.read(study.ROOT / "configs/r123_expansion_v1/scenes.json")["scenarios"][3]
        alias = copy.deepcopy(original)
        alias["obstacles"].reverse()
        alias["obstacle_layout_sha256"] = study.digest(alias["obstacles"])
        alias["scenario_id"] = "another-arm"
        alias["scenario_seed"] += 1
        actual, other = study.scene_identity(original), study.scene_identity(alias)
        for key in ("physical", "layout", "launch_positions", "motion_origin"):
            self.assertEqual(actual[key], other[key])
        alias["obstacle_layout_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "layout"):
            study.scene_identity(alias)

    def test_private_and_paired_stream_namespace(self):
        streams = study.read(study.CONFIG)["random_streams"]
        family = "physical-family"
        first = study.stream_seed(streams, family, "original_offer", 0)
        self.assertEqual(first, study.stream_seed(streams, family, "original_offer", 0))
        seeds = [study.stream_seed(streams, family, kind, boat)
                 for kind in ("original_offer", "extra_offer", "continuation") for boat in range(3)]
        self.assertEqual(len(set(seeds)), 9)
        self.assertNotIn(study.stream_seed(streams, family, "continuation", 1, 2), seeds)
        with self.assertRaises(ValueError):
            study.stream_seed(streams, family, "original_offer", 1, 2)

    def test_concurrent_repeat_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            value = {"schema": "one", "value": [1, 2]}
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = pool.map(lambda _: study.publish(directory, value), range(2))
            self.assertEqual(first, second)
            before = first.read_bytes()
            with self.assertRaises(FileExistsError):
                study.publish(directory, {"schema": "different"})
            self.assertEqual(first.read_bytes(), before)
            self.assertEqual(list(directory.iterdir()), [first])


class FrozenRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = study.read(study.ROOT / "verification/launch_assignment_v1/manifest.json")

    def test_no_cross_split_physical_key_and_no_missing_history(self):
        seen = {}
        expansion = set()
        actual_members = set()
        for family in self.manifest["families"]:
            for member in family["members"]:
                for key in study.FAMILY_KEYS:
                    previous = seen.setdefault((key, member[key]), family["family_id"])
                    self.assertEqual(previous, family["family_id"])
                if member["source"] != "fresh":
                    self.assertEqual(family["split"], "development")
                    actual_members.add((member["source"], member["index"], member["physical"]))
                if member["source"] == "expansion24":
                    expansion.add(member["scenario_id"])
        history = study.read(study.ROOT / self.manifest["study"]["historical_registry"])
        expected = {(s["name"], i, row["physical"]) for s in history["sources"]
                    for i, row in enumerate(s["scenes"])}
        self.assertEqual(actual_members, expected)
        self.assertEqual(expansion, {f"r123_expand_20261119_{i:04d}" for i in range(24)})
        quotas = self.manifest["study"]["native_inputs"]["fresh_families"]
        for split, count in quotas.items():
            fresh = [f for f in self.manifest["families"] if f["split"] == split
                     and all(m["source"] == "fresh" for m in f["members"])]
            self.assertEqual(len(fresh), count)

    def test_held_out_cell_and_final_families_are_sealed(self):
        cells = self.manifest["cells"]
        self.assertEqual(len(cells), 4)
        for cell in cells:
            self.assertEqual(cell["access"]["blind"], "sealed")
            held = (cell["search"], cell["terminal"]) == ("handoff_value", "direct")
            self.assertEqual(cell["access"]["development"], "sealed" if held else "fit")
            self.assertEqual(cell["access"]["calibration"], "sealed" if held else "calibrate_only")
        for family in self.manifest["families"]:
            for member in family["members"]:
                self.assertEqual(set(member), set(study.FAMILY_KEYS) | {"scenario_id", "source", "index"})
        self.assertEqual(set(self.manifest["study"]["budget"]["prepare"].values()), {0})


if __name__ == "__main__":
    unittest.main()
