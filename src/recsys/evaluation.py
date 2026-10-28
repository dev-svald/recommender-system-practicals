"""
Evaluation metrics for recommender systems.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range  # type: ignore


@njit
def evaluate_rmse(
    user_vals: np.ndarray,
    user_idx: np.ndarray,
    user_map: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray
) -> Tuple[float, float, int]:
    """
    Calculate Root Mean Squared Error (RMSE) for bias-only predictions.

    Parameters
    ----------
    user_vals : np.ndarray
        Flattened rating values.
    user_idx : np.ndarray
        Flattened item indices.
    user_map : np.ndarray (M, 2)
        Pointers for user rating boundaries.
    user_biases : np.ndarray (M,)
        User bias parameters.
    item_biases : np.ndarray (N,)
        Item bias parameters.

    Returns
    -------
    tuple of (rmse, sse, count)
    """
    M = user_map.shape[0]
    sse = 0.0
    count = 0

    for user_id in range(M):
        start_idx = user_map[user_id, 0]
        end_idx = user_map[user_id, 1]

        for ptr in range(start_idx, end_idx):
            movie_id = user_idx[ptr]
            rating = user_vals[ptr]

            prediction = user_biases[user_id] + item_biases[movie_id]
            error = rating - prediction
            sse += error * error
            count += 1

    rmse = float(np.sqrt(sse / count)) if count > 0 else float("inf")
    return rmse, sse, count


@njit(parallel=True)
def evaluate_rmse_m_e(
    vals: np.ndarray,
    items: np.ndarray,
    idx_map: np.ndarray,
    M: int,
    user_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_biases: np.ndarray,
    item_vectors: np.ndarray
) -> float:
    """
    Calculate RMSE for Matrix Factorization predictions.

    Prediction formula:
        r_pred = b_u + b_i + (u_u . v_i)
    """
    sse = 0.0
    count = 0
    K = user_vectors.shape[1]

    for user_idx in prange(M):
        start = idx_map[user_idx, 0]
        end = idx_map[user_idx, 1]

        if start == end:
            continue

        for i in range(start, end):
            movie_idx = items[i]
            rating = vals[i]

            dot = 0.0
            for k in range(K):
                dot += user_vectors[user_idx, k] * item_vectors[movie_idx, k]

            prediction = user_biases[user_idx] + item_biases[movie_idx] + dot
            error = rating - prediction
            sse += error * error
            count += 1

    return float(np.sqrt(sse / count)) if count > 0 else 0.0


def compute_mf_loss(
    sse: float,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    lambd: float,
    gamma: float,
    tau: float
) -> float:
    """
    Calculate the regularized objective loss for Matrix Factorization with Biases:
        L = 0.5 * lambda * SSE + 0.5 * gamma * (||b_u||^2 + ||b_i||^2) + 0.5 * tau * (||U||_F^2 + ||V||_F^2)
    """
    bias_reg = 0.5 * gamma * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
    emb_reg = 0.5 * tau * (np.sum(user_vectors ** 2) + np.sum(item_vectors ** 2))
    return float(0.5 * lambd * sse + bias_reg + emb_reg)
