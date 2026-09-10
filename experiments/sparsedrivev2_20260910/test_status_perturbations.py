"""CPU tests for the statistical and row-identity intervention contract."""
import math
import unittest

import numpy as np

from status_perturbations import (Condition, SESSION_VARIANCE_FRACTION,
                                 TAU_SECONDS, build_conditions, perturb_status,
                                 status_errors)


class StatusPerturbationsTest(unittest.TestCase):
    def setUp(self):
        self.sessions = np.array(["s1", "s2", "s1", "s2", "s1", "s2"])
        self.times = np.array([0., .1, .2, 1., 2., 3.])
        self.status = np.zeros((6, 8), dtype=np.float32)
        self.status[:, 4:] = [[1, .1, .3, -.1], [2, -.2, -.4, .2], [3, .3, .5, -.3],
                              [4, -.4, -.6, .4], [5, .5, .7, -.5], [6, -.6, -.8, .6]]
        self.condition = Condition("test_noise", "session_ou", vx_mae=.3, ax_mae=.1)

    def test_controls_and_bias_units_and_input_immutability(self):
        before = self.status.copy()
        conditions = build_conditions()
        self.assertEqual(len(conditions), 24)
        self.assertEqual(len({c.name for c in conditions}), 24)
        for condition in conditions:
            actual = perturb_status(self.status, self.sessions, self.times, condition)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.isfinite(actual).all())
            if condition.kind == "baseline":
                np.testing.assert_array_equal(actual, before)
                self.assertFalse(np.shares_memory(actual, self.status))
            elif condition.kind == "zero_status":
                np.testing.assert_array_equal(actual, np.zeros_like(before))
            else:
                np.testing.assert_array_equal(actual[:, [0, 1, 2, 3, 5, 7]], before[:, [0, 1, 2, 3, 5, 7]])
                if condition.kind == "bias":
                    np.testing.assert_allclose(actual[:, 4]-before[:, 4], condition.vx_bias, atol=3e-7)
                    np.testing.assert_allclose(actual[:, 6]-before[:, 6], condition.ax_bias, atol=1e-7)
        np.testing.assert_array_equal(self.status, before)

    def test_noise_is_bitwise_invariant_to_batch_subset_order_and_duplicates(self):
        reference = perturb_status(self.status, self.sessions, self.times, self.condition)
        for ids in (np.array([5, 0, 3, 2, 4, 1]), np.array([4, 1]), np.array([2, 2, 0])):
            actual = perturb_status(self.status[ids], self.sessions[ids], self.times[ids], self.condition)
            np.testing.assert_array_equal(actual, reference[ids])
        chunked = np.concatenate([perturb_status(self.status[i:i+1], self.sessions[i:i+1],
                                                self.times[i:i+1], self.condition) for i in range(6)])
        np.testing.assert_array_equal(reference, chunked)
        # Adding a much later observation cannot change the already-requested prefix.
        extended = status_errors(["s1", "s1", "s1"], [0., .2, 50.], self.condition)
        original = status_errors(["s1", "s1"], [0., .2], self.condition)
        np.testing.assert_array_equal(extended[:2], original)

    def test_strengths_share_draws_without_sample_centering_or_status_dependency(self):
        c1 = Condition("low", "session_ou", vx_mae=.1, ax_mae=.1)
        c2 = Condition("high", "session_ou", vx_mae=.3, ax_mae=.3)
        low = status_errors(self.sessions, self.times, c1)
        high = status_errors(self.sessions, self.times, c2)
        np.testing.assert_allclose(high, 3*low, atol=2e-16, rtol=1e-15)
        one = status_errors(["s1"], [.2], c1)
        self.assertNotEqual(float(one[0, 4]), 0.)  # A one-row sample was not centered.
        self.assertNotAlmostEqual(float(np.abs(one[:, 4]).mean()), .1)
        other_status = self.status + np.array([0, 0, 0, 0, 5, 0, -2, 0], dtype=np.float32)
        first = perturb_status(self.status, self.sessions, self.times, c1)
        second = perturb_status(other_status, self.sessions, self.times, c1)
        np.testing.assert_allclose(second[:, [4, 6]]-first[:, [4, 6]],
                                   np.tile([5, -2], (6, 1)), atol=1e-6)
        self.assertFalse(np.array_equal(status_errors(self.sessions, self.times, c1, seed=1), low))

    def test_stationary_marginal_mae_and_session_time_covariance(self):
        # Independent sessions supply independent replicates; this checks the
        # population law, not a long single-session path dominated by its bias.
        n = 5000
        names = np.repeat([f"statistical_session_{i}" for i in range(n)], 3)
        times = np.tile([0., .1, 5.], n)
        errors = status_errors(names, times, self.condition).reshape(n, 3, 8)
        sigma = .3 * math.sqrt(math.pi/2)
        standardized = errors[:, :, 4] / sigma
        np.testing.assert_allclose(np.abs(errors[:, :, 4]).mean(0), .3, rtol=.04)
        np.testing.assert_allclose(np.abs(errors[:, :, 6]).mean(0), .1, rtol=.04)
        self.assertTrue((np.abs(standardized.mean(0)) < .05).all())
        np.testing.assert_allclose(standardized.var(0), 1., atol=.07)
        cov = np.cov(standardized.T, ddof=0)
        for column, lag in ((1, .1), (2, 5.)):
            expected = SESSION_VARIANCE_FRACTION + (1-SESSION_VARIANCE_FRACTION)*math.exp(-lag/TAU_SECONDS)
            self.assertAlmostEqual(float(cov[0, column]), expected, delta=.06)
        # Axis-specific seeds must not spuriously couple vx and ax errors.
        self.assertLess(abs(float(np.corrcoef(errors[:, 0, 4], errors[:, 0, 6])[0, 1])), .05)
        self.assertGreater(float(cov[0, 1]), float(cov[0, 2]) + .3)

    def test_invalid_full_status_metadata_and_condition_are_rejected(self):
        for column in range(8):
            broken = self.status.copy(); broken[0, column] = np.nan
            with self.assertRaises(ValueError):
                perturb_status(broken, self.sessions, self.times, self.condition)
        broken = self.status.copy(); broken[0, 2] = 1.
        with self.assertRaises(ValueError):
            perturb_status(broken, self.sessions, self.times, self.condition)
        for times in ([0., .15], [-.1, 0.], [0., np.inf], [0., 1700000000.]):
            with self.assertRaises(ValueError):
                status_errors(["a", "b"], times, self.condition)
        with self.assertRaises(ValueError):
            status_errors(["a"], [0., .1], self.condition)
        with self.assertRaises(ValueError):
            status_errors([""], [0.], self.condition)
        with self.assertRaises(ValueError):
            status_errors(["a"], [0.], Condition("zero", "zero_status"))
        with self.assertRaises(ValueError):
            Condition("bad", "session_ou", vx_bias=.1, vx_mae=.3)
        with self.assertRaises(ValueError):
            Condition("bad", "session_ou", vx_mae=-1.)
        with self.assertRaises(ValueError):
            perturb_status(self.status[:1], self.sessions, self.times, self.condition)
        with self.assertRaises(ValueError):
            perturb_status(self.status, self.sessions, self.times, self.condition, seed=True)

    def test_empty_population_is_well_defined(self):
        out = perturb_status(np.empty((0, 8), np.float32), np.array([], dtype=str), [], self.condition)
        self.assertEqual(out.shape, (0, 8))

    def test_unmodified_float32_slots_preserve_signed_zero_bits(self):
        status = np.full((2, 8), -0., np.float32)
        baseline = perturb_status(status, ["a", "a"], [0., .1], Condition("base", "baseline"))
        self.assertEqual(status.tobytes(), baseline.tobytes())
        biased = perturb_status(status, ["a", "a"], [0., .1], Condition("bias", "bias", ax_bias=.3))
        unchanged = [0, 1, 2, 3, 4, 5, 7]
        self.assertEqual(status[:, unchanged].tobytes(), biased[:, unchanged].tobytes())


if __name__ == "__main__":
    unittest.main()
