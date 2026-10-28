"""
Unit tests for ALS models (Baseline Bias and Matrix Factorization).
"""

import unittest
from recsys.data import build_indices
from recsys.models.baseline import BaselineBiasModel
from recsys.models.matrix_factorization import MatrixFactorizationALS
from recsys.recommend import create_dummy_user, recommend_movies


class TestRecsysModels(unittest.TestCase):
    def setUp(self):
        ratings = [
            (1, 10, 5.0),
            (1, 20, 4.0),
            (1, 30, 1.0),
            (2, 10, 4.5),
            (2, 20, 4.0),
            (2, 30, 2.0),
            (3, 10, 1.0),
            (3, 20, 2.0),
            (3, 30, 5.0),
            (4, 10, 2.0),
            (4, 20, 1.0),
            (4, 30, 4.5),
        ]
        (_, _, self.data_u), (_, _, self.data_m) = build_indices(ratings)
        self.M = len(self.data_u)
        self.N = len(self.data_m)

    def test_baseline_bias_training(self):
        model = BaselineBiasModel(lambd=0.1, gamma=0.1, num_iters=5)
        model.fit(self.data_u, self.data_m, self.data_u, self.M, self.N, verbose=False)

        self.assertIsNotNone(model.user_biases)
        self.assertIsNotNone(model.item_biases)
        self.assertEqual(len(model.user_biases), self.M)
        self.assertEqual(len(model.item_biases), self.N)
        self.assertEqual(len(model.history["loss"]), 5)

    def test_matrix_factorization_training(self):
        model = MatrixFactorizationALS(K=3, lambd=0.1, gamma=0.1, tau=0.5, num_iters=5, random_seed=42)
        model.fit(self.data_u, self.data_m, self.data_u, self.M, self.N, verbose=False)

        self.assertIsNotNone(model.user_vectors)
        self.assertIsNotNone(model.item_vectors)
        self.assertEqual(model.user_vectors.shape, (self.M, 3))
        self.assertEqual(model.item_vectors.shape, (self.N, 3))

    def test_recommendation_flow(self):
        model = MatrixFactorizationALS(K=4, lambd=0.1, gamma=0.1, tau=0.5, num_iters=5, random_seed=42)
        model.fit(self.data_u, self.data_m, self.data_u, self.M, self.N, verbose=False)

        idx_to_movieid = [10, 20, 30]
        movie_id_to_title = {10: "Sci-Fi Epic", 20: "Space Odyssey", 30: "Romantic Comedy"}

        dummy_vec, loved_indices = create_dummy_user(
            loved_movie_titles=["Sci-Fi Epic"],
            user_vectors=model.user_vectors,
            item_vectors=model.item_vectors,
            item_biases=model.item_biases,
            idx_to_movieid=idx_to_movieid,
            movie_id_to_title=movie_id_to_title,
            K=4
        )

        self.assertEqual(len(dummy_vec), 4)
        self.assertEqual(loved_indices, [0])

        recs = recommend_movies(
            dummy_user_vector=dummy_vec,
            item_vectors=model.item_vectors,
            item_biases=model.item_biases,
            idx_to_movieid=idx_to_movieid,
            movie_id_to_title=movie_id_to_title,
            exclude_indices=loved_indices,
            top_k=2
        )

        self.assertEqual(len(recs), 2)
        self.assertNotEqual(recs[0]["movie_idx"], 0)


if __name__ == "__main__":
    unittest.main()
