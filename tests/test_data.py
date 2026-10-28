"""
Unit tests for recsys.data module.
"""

import unittest
from recsys.data import build_indices, split_train_test, prepare_flat_data


class TestRecsysData(unittest.TestCase):
    def setUp(self):
        self.sample_ratings = [
            (101, 201, 4.0),
            (101, 202, 5.0),
            (102, 201, 3.0),
            (102, 203, 2.0),
            (103, 202, 4.5),
            (103, 203, 1.0),
        ]

    def test_build_indices(self):
        (u_map, idx_to_u, data_u), (m_map, idx_to_m, data_m) = build_indices(self.sample_ratings)

        self.assertEqual(len(u_map), 3)
        self.assertEqual(len(idx_to_u), 3)
        self.assertEqual(len(m_map), 3)
        self.assertEqual(len(idx_to_m), 3)
        self.assertEqual(len(data_u), 3)
        self.assertEqual(len(data_m), 3)
        self.assertEqual(len(data_u[0]), 2)

    def test_split_train_test(self):
        (_, _, data_u), _ = build_indices(self.sample_ratings)
        extended_data_u = [r * 5 for r in data_u]

        u_tr, u_te, m_tr, m_te = split_train_test(
            extended_data_u, number_of_users=3, number_of_movies=3, train_ratio=0.8, verbose=False
        )

        self.assertEqual(len(u_tr), 3)
        self.assertEqual(len(u_te), 3)
        total_tr = sum(len(r) for r in u_tr)
        total_te = sum(len(r) for r in u_te)
        self.assertEqual(total_tr + total_te, sum(len(r) for r in extended_data_u))

    def test_prepare_flat_data(self):
        (_, _, data_u), (_, _, data_m) = build_indices(self.sample_ratings)
        M = len(data_u)
        N = len(data_m)

        u_vals, u_idx, u_map, i_vals, i_idx, i_map = prepare_flat_data(data_u, data_m, M, N)

        self.assertEqual(len(u_vals), 6)
        self.assertEqual(len(u_idx), 6)
        self.assertEqual(u_map.shape, (3, 2))
        self.assertEqual(len(i_vals), 6)
        self.assertEqual(len(i_idx), 6)
        self.assertEqual(i_map.shape, (3, 2))
        self.assertEqual(u_map[0, 1] - u_map[0, 0], 2)


if __name__ == "__main__":
    unittest.main()
