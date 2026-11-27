import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from defectclip.data import find_mvtec_root, load_mvtec
from defectclip.localize import _pool_windows
from defectclip.prompts import defect_phrase, retrieval_prompts, state_prompts


class PromptTest(unittest.TestCase):
    def test_ensemble_size_and_phrases(self):
        normal, anomaly = state_prompts("metal_nut", variant="ensemble")
        self.assertEqual(len(normal), 36)
        self.assertEqual(len(anomaly), 36)
        self.assertTrue(all("metal nut" in p for p in normal))
        self.assertEqual(defect_phrase("broken_large"), "large break")
        queries = retrieval_prompts("carpet", "hole")
        self.assertEqual(len(queries), 4)
        self.assertIn("hole", queries[0])

    def test_defect_vocab_adds_names(self):
        _normal, anomaly = state_prompts("capsule", variant="defect_vocab", defect_names=["crack", "scratch"])
        self.assertGreater(len(anomaly), 36)
        self.assertTrue(any("crack" in p for p in anomaly))

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            state_prompts("bottle", variant="nope")


class DataTest(unittest.TestCase):
    def test_loader_reads_a_tiny_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "bottle" / "test" / "broken_large" / "000.png"
            mask = root / "bottle" / "ground_truth" / "broken_large" / "000_mask.png"
            good = root / "bottle" / "train" / "good" / "000.png"
            image.parent.mkdir(parents=True)
            mask.parent.mkdir(parents=True)
            good.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), (255, 0, 0)).save(image)
            Image.new("L", (8, 8), 255).save(mask)
            Image.new("RGB", (8, 8), (0, 255, 0)).save(good)
            # The loader expects every requested category. Ask only for bottle
            # by pointing find at a root that has bottle/train/good.
            found = find_mvtec_root(root)
            self.assertEqual(found, root)
            records = load_mvtec(root, categories=["bottle"])
            self.assertEqual(len(records), 2)
            test = [r for r in records if r.split == "test"]
            self.assertEqual(test[0].defect, "broken_large")
            self.assertTrue(test[0].is_anomaly)
            self.assertTrue(test[0].mask_path.endswith("000_mask.png"))


class WindowTest(unittest.TestCase):
    def test_pool_keeps_shape(self):
        feat = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        self.assertEqual(_pool_windows(feat, 1).shape, (2, 2, 3))
        pooled = _pool_windows(feat, 3)
        self.assertEqual(pooled.shape, (2, 2, 3))
        # Edge padding plus a window larger than the map mixes every cell into both corners.
        self.assertTrue(np.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()
