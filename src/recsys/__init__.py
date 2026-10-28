"""
Recommender Systems at Scale.

A high-performance implementation of Alternating Least Squares (ALS) Matrix Factorization
with User/Item Biases, Numba JIT multi-core parallelization, and cold-start recommendation inference.
"""

from recsys.data import (
    build_indices,
    compute_movie_rating_counts,
    load_movie_metadata,
    prepare_flat_data,
    split_train_test,
)
from recsys.evaluation import evaluate_rmse, evaluate_rmse_m_e, compute_mf_loss
from recsys.models.baseline import BaselineBiasModel, train_bias_model
from recsys.models.matrix_factorization import (
    MatrixFactorizationALS,
    load_mf_model,
    save_mf_model,
    train_model_m_e,
)
from recsys.recommend import (
    create_dummy_user,
    display_recommendations,
    find_non_polarizing_movies,
    find_polarizing_movies,
    recommend_movies,
)
from recsys.tuning import grid_search_bias_hyperparameters
from recsys.visualization import plot_training_curves

__version__ = "1.0.0"

__all__ = [
    "build_indices",
    "compute_movie_rating_counts",
    "load_movie_metadata",
    "prepare_flat_data",
    "split_train_test",
    "evaluate_rmse",
    "evaluate_rmse_m_e",
    "compute_mf_loss",
    "BaselineBiasModel",
    "train_bias_model",
    "MatrixFactorizationALS",
    "train_model_m_e",
    "save_mf_model",
    "load_mf_model",
    "create_dummy_user",
    "recommend_movies",
    "display_recommendations",
    "find_polarizing_movies",
    "find_non_polarizing_movies",
    "grid_search_bias_hyperparameters",
    "plot_training_curves",
]
