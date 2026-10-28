"""
Baseline Recommender Model based on User and Item Biases optimized via ALS.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
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

from recsys.data import prepare_flat_data
from recsys.evaluation import evaluate_rmse


@njit(parallel=True)
def update_user_biases(
    user_vals: np.ndarray,
    user_idx: np.ndarray,
    user_map: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    lambd: float,
    gamma: float
) -> None:
    """
    Update all user biases in parallel using Alternating Least Squares (ALS).

    Closed-form solution for user u:
        b_u = sum_i lambda * (r_ui - b_i) / (lambda * |R_u| + gamma)
    """
    M = user_map.shape[0]

    for user_id in prange(M):
        start_idx = user_map[user_id, 0]
        end_idx = user_map[user_id, 1]

        numerator = 0.0
        denominator = 0.0

        for ptr in range(start_idx, end_idx):
            movie_id = user_idx[ptr]
            rating = user_vals[ptr]

            residual = rating - item_biases[movie_id]
            numerator += lambd * residual
            denominator += lambd

        if denominator > 0:
            user_biases[user_id] = numerator / (denominator + gamma)


@njit(parallel=True)
def update_item_biases(
    item_vals: np.ndarray,
    item_idx: np.ndarray,
    item_map: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    lambd: float,
    gamma: float
) -> None:
    """
    Update all item biases in parallel using Alternating Least Squares (ALS).

    Closed-form solution for item i:
        b_i = sum_u lambda * (r_ui - b_u) / (lambda * |R_i| + gamma)
    """
    N = item_map.shape[0]

    for movie_id in prange(N):
        start_idx = item_map[movie_id, 0]
        end_idx = item_map[movie_id, 1]

        numerator = 0.0
        denominator = 0.0

        for ptr in range(start_idx, end_idx):
            user_id = item_idx[ptr]
            rating = item_vals[ptr]

            residual = rating - user_biases[user_id]
            numerator += lambd * residual
            denominator += lambd

        if denominator > 0:
            item_biases[movie_id] = numerator / (denominator + gamma)


def predict_rating(
    user_idx: int,
    movie_idx: int,
    user_biases: np.ndarray,
    item_biases: np.ndarray
) -> float:
    """
    Predict rating for a given (user_idx, movie_idx) pair.
    """
    b_u = user_biases[user_idx] if user_idx < len(user_biases) else 0.0
    b_i = item_biases[movie_idx] if movie_idx < len(item_biases) else 0.0
    return float(b_u + b_i)


def train_bias_model(
    data_by_user_train: List[List[Tuple[int, float]]],
    data_by_movie_train: List[List[Tuple[int, float]]],
    data_by_user_test: List[List[Tuple[int, float]]],
    M: int,
    N: int,
    lambd: float = 0.1,
    gamma: float = 0.1,
    num_iters: int = 30,
    verbose: bool = True
) -> Dict[str, Union[np.ndarray, List[float]]]:
    """
    Train bias-only model using parallel ALS.
    """
    if verbose:
        print("Preparing contiguous flat arrays for ALS...")

    u_vals_tr, u_idx_tr, u_map_tr, i_vals_tr, i_idx_tr, i_map_tr = prepare_flat_data(
        data_by_user_train, data_by_movie_train, M, N
    )
    u_vals_te, u_idx_te, u_map_te, _, _, _ = prepare_flat_data(
        data_by_user_test, [[] for _ in range(N)], M, N
    )

    user_biases = np.zeros(M, dtype=np.float64)
    item_biases = np.zeros(N, dtype=np.float64)

    loss_history: List[float] = []
    train_rmse_history: List[float] = []
    test_rmse_history: List[float] = []

    if verbose:
        print("=" * 60)
        print(f"Starting Baseline Bias ALS Training (lambda={lambd}, gamma={gamma}, iters={num_iters})")
        print(f"Users: {M:,} | Movies: {N:,}")
        print("=" * 60)

    for it in range(num_iters):
        update_user_biases(u_vals_tr, u_idx_tr, u_map_tr, user_biases, item_biases, lambd, gamma)
        update_item_biases(i_vals_tr, i_idx_tr, i_map_tr, user_biases, item_biases, lambd, gamma)

        train_rmse, train_sse, _ = evaluate_rmse(u_vals_tr, u_idx_tr, u_map_tr, user_biases, item_biases)
        test_rmse, _, _ = evaluate_rmse(u_vals_te, u_idx_te, u_map_te, user_biases, item_biases)

        loss_history.append(train_sse)
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)

        if verbose:
            print(f"Iter {it + 1:2d}/{num_iters:2d} | Loss: {train_sse:,.0f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    if verbose:
        print("=" * 60)
        print(f"Final Train RMSE: {train_rmse_history[-1]:.4f} | Final Test RMSE: {test_rmse_history[-1]:.4f}\n")

    return {
        "user_biases": user_biases,
        "item_biases": item_biases,
        "loss_history": loss_history,
        "train_rmse_history": train_rmse_history,
        "test_rmse_history": test_rmse_history,
    }


class BaselineBiasModel:
    """
    Object-oriented wrapper for the User/Item Bias Recommender model.
    """

    def __init__(self, lambd: float = 0.1, gamma: float = 0.1, num_iters: int = 30):
        self.lambd = lambd
        self.gamma = gamma
        self.num_iters = num_iters
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.history: Dict[str, List[float]] = {}

    def fit(
        self,
        data_by_user_train: List[List[Tuple[int, float]]],
        data_by_movie_train: List[List[Tuple[int, float]]],
        data_by_user_test: List[List[Tuple[int, float]]],
        M: int,
        N: int,
        verbose: bool = True
    ) -> "BaselineBiasModel":
        results = train_bias_model(
            data_by_user_train=data_by_user_train,
            data_by_movie_train=data_by_movie_train,
            data_by_user_test=data_by_user_test,
            M=M,
            N=N,
            lambd=self.lambd,
            gamma=self.gamma,
            num_iters=self.num_iters,
            verbose=verbose
        )
        self.user_biases = results["user_biases"]
        self.item_biases = results["item_biases"]
        self.history = {
            "loss": results["loss_history"],
            "train_rmse": results["train_rmse_history"],
            "test_rmse": results["test_rmse_history"],
        }
        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        if self.user_biases is None or self.item_biases is None:
            raise ValueError("Model is not fitted yet.")
        return predict_rating(user_idx, movie_idx, self.user_biases, self.item_biases)
