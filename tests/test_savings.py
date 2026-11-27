import unittest

from defectclip.savings import estimate_labor_delta


class SavingsTest(unittest.TestCase):
    def test_formula(self):
        # 3600 parts/hour, 10 manual seconds each, $36/hour, 10 images/second.
        # Manual labor is 10 hours per clock hour. The model needs 0.1 hour.
        estimate = estimate_labor_delta(3600, 10, 36, 10)
        self.assertAlmostEqual(estimate["manual_labor_hours_per_clock_hour"], 10.0)
        self.assertAlmostEqual(estimate["model_seconds_per_part"], 0.1)
        self.assertAlmostEqual(estimate["model_compute_hours_per_clock_hour"], 0.1)
        self.assertAlmostEqual(estimate["hours_difference_per_clock_hour"], 9.9)
        self.assertAlmostEqual(estimate["labor_cost_difference_usd_per_clock_hour"], 9.9 * 36)
        self.assertIn("not a measured saving", estimate["kind"])

    def test_model_slower_than_the_person_is_a_negative_delta(self):
        estimate = estimate_labor_delta(10, 1, 20, 0.5)
        self.assertLess(estimate["hours_difference_per_clock_hour"], 0)

    def test_rejects_a_non_positive_throughput(self):
        with self.assertRaises(ValueError):
            estimate_labor_delta(10, 1, 20, 0)


if __name__ == "__main__":
    unittest.main()
