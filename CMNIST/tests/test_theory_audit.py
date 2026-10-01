import unittest

import numpy as np

from analyze_tail_support import weighted_cvar
from protocol import canonical_condition, deployment_epsilon, discrete_cvar
from priority7_theory import cvar, deployment_prior, one_trial, risk_profiles, source_prior


class TheoryAuditTests(unittest.TestCase):
    def test_synthetic_uniform_deployment_gap_is_fixed(self):
        profiles = risk_profiles(10, 0.5)
        deployment = deployment_prior(10)

        for alpha in [0.5, 0.75, 0.9]:
            gap = cvar(profiles["head_favored"], deployment, alpha) - cvar(profiles["tail_favored"], deployment, alpha)
            self.assertLess(abs(gap - 0.05), 1e-12)

    def test_synthetic_source_prior_removes_final_tail_domains(self):
        source, missing_count = source_prior(10, exponent=1.0, missing_tail_fraction=0.3)
        expected = np.arange(1, 11, dtype=float) ** -1.0
        expected[7:] = 0.0
        expected /= expected.sum()

        self.assertEqual(missing_count, 3)
        self.assertEqual(np.count_nonzero(source), 7)
        self.assertTrue(np.allclose(source, expected))

    def test_synthetic_reversal_condition_orientation(self):
        result = one_trial(
            n_domains=10,
            sample_size=100,
            exponent=0.0,
            missing_tail_fraction=0.3,
            alpha=0.9,
            tradeoff=0.5,
            rng=np.random.default_rng(0),
        )

        self.assertEqual(result["deployment_winner"], "tail_favored")
        self.assertEqual(result["empirical_winner"], "head_favored")
        self.assertEqual(result["reversal"], 1)

    def test_weighted_discrete_cvar_uses_fractional_boundary_mass(self):
        self.assertLess(abs(cvar([0, 1], [0.9, 0.1], 0.9) - 1.0), 1e-12)
        self.assertLess(abs(cvar([0, 10], [0.75, 0.25], 0.5) - 5.0), 1e-12)
        self.assertLess(abs(weighted_cvar([0, 10], [0.75, 0.25], 0.5) - 5.0), 1e-12)

    def test_canonical_conditions_and_epsilon(self):
        self.assertEqual(canonical_condition("near_missing_tail"), "scarce_tail")
        self.assertEqual(deployment_epsilon("scarce_tail"), 0.0)
        self.assertEqual(deployment_epsilon("missing_tail"), 0.25)

    def test_discrete_cvar_rejects_invalid_weights(self):
        with self.assertRaises(ValueError):
            discrete_cvar([0.0, 1.0], [0.5, 0.6], 0.5)
        with self.assertRaises(ValueError):
            discrete_cvar([0.0, 1.0], [1.1, -0.1], 0.5)
        with self.assertRaises(ValueError):
            discrete_cvar([], [], 0.5)


if __name__ == "__main__":
    unittest.main()