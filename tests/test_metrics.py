import unittest

import numpy as np

from defectclip.metrics import average_precision, hit_at_k, rank_relevance, recall_at_k, safe_auroc


class MetricsTest(unittest.TestCase):
    def test_ranking_recall_and_hit(self):
        relevance = np.array([1, 0, 1, 0])
        scores = np.array([0.2, 0.9, 0.8, 0.1])
        ranked = rank_relevance(relevance, scores)
        self.assertEqual(ranked.tolist(), [False, True, True, False])
        self.assertEqual(recall_at_k(ranked, 1), 0.0)
        self.assertAlmostEqual(recall_at_k(ranked, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranked, 3), 1.0)
        self.assertEqual(hit_at_k(ranked, 1), 0.0)
        self.assertEqual(hit_at_k(ranked, 2), 1.0)

    def test_average_precision(self):
        relevance = np.array([1, 0, 1, 0])
        scores = np.array([0.2, 0.9, 0.8, 0.1])
        ap = average_precision(relevance, scores)
        self.assertAlmostEqual(ap, (0.5 + (2.0 / 3.0)) / 2.0)

    def test_auroc_perfect_and_empty(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.1, 0.8, 0.9])
        self.assertEqual(safe_auroc(labels, scores), 1.0)
        self.assertIsNone(safe_auroc(np.array([1, 1, 1]), scores[:3]))


if __name__ == "__main__":
    unittest.main()
