"""Synthetic CPU checks only: no held arrays, identities or models are loaded."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('confirmation_analysis', Path(__file__).with_name('analyze_confirmation_result.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def arrays(head, base):
    head = np.asarray(head, np.float64)
    base = np.asarray(base, np.float64)
    n = len(head)
    out = {'rows': np.arange(n, dtype=np.int64), 'shortlist_oracle': np.zeros(n)}
    for prefix, point in [('', head), ('base_', base)]:
        xy = np.stack((point, np.zeros_like(point)), -1)
        out.update({prefix+'pred': xy.copy(), prefix+'error_xy': xy.copy(), prefix+'point_l2': point,
                    prefix+'d3': point @ m.WEIGHTS, prefix+'candidate_id': np.arange(n, dtype=np.int64) + bool(prefix)})
    return out


class ConfirmationAnalysisContracts(unittest.TestCase):
    def test_metric_matches_three_prefix_ades_and_cluster_constant_shift(self):
        head = np.tile(np.arange(1., 7.), (6, 1))
        a = arrays(head, head + .2)
        summary = m.summarize(a, ['a', 'a', 'a', 'b', 'b', 'c'])
        explicit = np.mean([head[:, :2].mean(), head[:, :4].mean(), head.mean()])
        self.assertAlmostEqual(summary['methods']['head']['d3'], explicit)
        self.assertAlmostEqual(explicit, 2.5)
        np.testing.assert_allclose(summary['head_minus_base']['d3_95ci'], [-.2, -.2], atol=1e-14)
        np.testing.assert_allclose(summary['head_minus_base']['point_l2_95ci'], np.full((2, 6), -.2), atol=1e-14)
        self.assertEqual(summary['head_minus_base']['improved_sessions'], 3)
        self.assertEqual(summary['head_minus_base']['improved_rows'], 6)
        self.assertEqual(summary['sessions']['a']['rows'], 3)

    def test_zero_difference_and_shared_candidates_do_not_invent_improvement(self):
        a = arrays(np.ones((4, 6)), np.ones((4, 6)))
        a['base_candidate_id'] = a['candidate_id'].copy()
        summary = m.summarize(a, ['one', 'one', 'two', 'two'])
        self.assertEqual(summary['head_minus_base']['changed_candidate_ids'], 0)
        self.assertEqual(summary['head_minus_base']['improved_sessions'], 0)
        self.assertEqual(summary['head_minus_base']['equal_d3_rows'], 4)
        self.assertEqual(summary['head_minus_base']['d3_95ci'], [0., 0.])

    def test_mismatched_targets_or_corrupted_metric_fail(self):
        a = arrays(np.ones((4, 6)), np.ones((4, 6))*2)
        a['base_pred'] = a['base_pred'] + .1  # errors no longer imply the same GT
        with self.assertRaisesRegex(ValueError, 'different targets'):
            m.validate_arrays(a)
        a = arrays(np.ones((4, 6)), np.ones((4, 6))*2)
        a['d3'][0] += .1
        with self.assertRaisesRegex(ValueError, 'does not recompute'):
            m.validate_arrays(a)
        a = arrays(np.ones((4, 6)), np.ones((4, 6))*2)
        a['shortlist_oracle'][0] = 1.1
        with self.assertRaisesRegex(ValueError, 'Oracle exceeds'):
            m.validate_arrays(a)

    def test_duplicate_rows_and_nonfinite_values_fail(self):
        a = arrays(np.ones((4, 6)), np.ones((4, 6))*2)
        a['rows'][1] = a['rows'][0]
        with self.assertRaisesRegex(ValueError, 'sorted unique'):
            m.validate_arrays(a)
        a = arrays(np.ones((4, 6)), np.ones((4, 6))*2)
        a['pred'][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'numeric values'):
            m.validate_arrays(a)

    def test_receipt_rejects_primary_ancestry_batch_or_changed_head(self):
        receipt = {'status': 'completed', 'audit_only': False, 'population': 'confirmation12', 'n': 1728,
                   'held_evaluation_enabled': True, 'same_forward_base_comparison': True, 'same_candidate_set': True,
                   'evaluation_batch_size': 1, 'evaluation_precision': 'bf16_base_fp32_head',
                   'confirmation_binding': {'synthetic': True},
                   'base': {'rows_sha256': m.HELD_SHA, 'fit_rows_sha256': m.TRAIN_SHA, 'fit_rows': 46170,
                            'fit_scenes': 171, 'evaluation_scenes': 32, 'evaluation_sessions': 12,
                            'bank_sha256': m.BANK_SHA, 'checkpoint_sha256': 'base', 'step': 2000, 'goal_mode': 'none'},
                   'relative': {'bank_sha256': m.BANK_SHA, 'base_checkpoint_sha256': 'base',
                                'relative_checkpoint_sha256': 'head', 'relative_step': 2000,
                                'relative_objective': 'soft_ce', 'base_goal_mode': 'none', 'feature_goal_mode': 'selection'}}
        m.validate_receipt(receipt, 'head')
        with self.assertRaisesRegex(ValueError, 'preselected'):
            m.validate_receipt(receipt, 'otherhead')
        receipt['evaluation_batch_size'] = 8
        with self.assertRaisesRegex(ValueError, 'fixed B1'):
            m.validate_receipt(receipt, 'head')
        receipt['evaluation_batch_size'] = 1
        receipt['base']['fit_rows_sha256'] = 'primary203'
        with self.assertRaisesRegex(ValueError, 'ancestry'):
            m.validate_receipt(receipt, 'head')


if __name__ == '__main__':
    unittest.main()
