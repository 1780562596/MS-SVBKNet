import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.build_scene_split import select_validation_scenes
from tools.yolo_compare import matched_inputs, run_comparison


class ReleaseToolsTest(unittest.TestCase):
    def test_paper_split_is_18_train_and_four_validation_scenes(self):
        scenes = [f"scene_{i:02d}" for i in range(22)]
        chosen = select_validation_scenes(scenes)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(len(set(scenes) - chosen), 18)
        self.assertEqual(chosen, select_validation_scenes(list(reversed(scenes))))
        self.assertNotEqual(chosen, select_validation_scenes(scenes, seed=3408))

    def test_explicit_fraction_and_invalid_counts(self):
        scenes = list("abcdefghij")
        self.assertEqual(len(select_validation_scenes(scenes, val_fraction=0.2)), 2)
        for kwargs in ({"val_scenes": 0}, {"val_scenes": 10}, {"val_fraction": 1},
                       {"val_scenes": 2, "val_fraction": 0.2}):
            with self.assertRaises(ValueError):
                select_validation_scenes(scenes, **kwargs)

    def test_missing_corresponding_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            roots = {name: Path(folder) / name for name in ("blur", "restored", "sharp")}
            for root in roots.values():
                (root / "scene").mkdir(parents=True)
                (root / "scene" / "frame.png").touch()
            self.assertEqual(len(matched_inputs(roots)["sharp"]), 1)
            (roots["sharp"] / "scene" / "other.png").touch()
            with self.assertRaisesRegex(ValueError, "matching"):
                matched_inputs(roots)

    def test_detection_branches_receive_identical_explicit_protocol(self):
        class Detector:
            def __init__(self):
                self.predict_calls, self.val_calls = [], []
            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                return []
            def val(self, **kwargs):
                self.val_calls.append(kwargs)
                return SimpleNamespace(box=SimpleNamespace(map50=0.5, map=0.3, mp=0.6, mr=0.4))
        detector = Detector()
        inputs = {label: ["frame.png"] for label in ("blur", "restored", "sharp")}
        datasets = {label: label + ".yaml" for label in inputs}
        summary = run_comparison(detector, inputs, datasets, [2, 3, 5, 7])
        self.assertEqual(set(summary), set(inputs))
        self.assertEqual(len(detector.val_calls), 3)
        for call in detector.predict_calls:
            self.assertEqual((call["imgsz"], call["conf"], call["iou"]), (640, 0.25, 0.70))
        for call in detector.val_calls:
            self.assertEqual((call["imgsz"], call["conf"], call["iou"]), (640, 0.001, 0.70))


if __name__ == "__main__":
    unittest.main()
