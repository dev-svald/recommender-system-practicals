"""
Matrix Factorization with User and Item Biases optimized via Parallel ALS.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple, Union
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
from recsys.evaluation import evaluate_rmse_m_e


@njit(parallel=True)
def update_user_biases_m_e(
    user_vals: np.ndarray,
    user_items: np.ndarray,
    user_map: np.ndarray,
    M: int,
    user_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_biases: np.ndarray,
    item_vectors: np.ndarray,
    lambd: float,
    gamma: float
) -> None:
    """
    Update user biases in parallel using ALS.

    Formula:
        residual = r_ui - (b_i + u_u . v_i)
        b_u = sum_i lambda * residual / (lambda * |R_u| + gamma)
    """
    K = user_vectors.shape[1]

    for user_idx in prange(M):
        start = user_map[user_idx, 0]
        end = user_map[user_idx, 1]

        if start == end:
            continue

        bias_numerator = 0.0
        for i in range(start, end):
            movie_idx = user_items[i]
            rating = user_vals[i]

            dot = 0.0
            for k in range(K):
                dot += user_vectors[user_idx, k] * item_vectors[movie_idx, k]

            prediction_no_bias = item_biases[movie_idx] + dot
            residual = rating - prediction_no_bias
            bias_numerator += lambd * residual

        user_biases[user_idx] = bias_numerator / (lambd * (end - start) + gamma)


@njit(parallel=True)
def update_user_vectors_m_e(
    user_vals: np.ndarray,
    user_items: np.ndarray,
    user_map: np.ndarray,
    M: int,
    user_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_biases: np.ndarray,
    item_vectors: np.ndarray,
    lambd: float,
    tau: float
) -> None:
    """
    Update user latent embeddings in parallel by solving regularized ridge linear systems:
        (lambda * sum_i v_i v_i^T + tau * I) u_u = lambda * sum_i (r_ui - b_u - b_i) v_i
    """
    K = user_vectors.shape[1]

    for user_idx in prange(M):
        start = user_map[user_idx, 0]
        end = user_map[user_idx, 1]

        if start == end:
            continue

        A_matrix = np.zeros((K, K), dtype=np.float64)
        b_vector = np.zeros(K, dtype=np.float64)

        for i in range(start, end):
            movie_idx = user_items[i]
            rating = user_vals[i]
            residual = rating - user_biases[user_idx] - item_biases[movie_idx]
            item_vec = item_vectors[movie_idx]

            for k1 in range(K):
                for k2 in range(K):
                    A_matrix[k1, k2] += lambd * item_vec[k1] * item_vec[k2]
                b_vector[k1] += lambd * residual * item_vec[k1]

        for k in range(K):
            A_matrix[k, k] += tau

        user_vectors[user_idx] = np.linalg.solve(A_matrix, b_vector)


@njit(parallel=True)
def update_item_biases_m_e(
    item_vals: np.ndarray,
    item_users: np.ndarray,
    item_map: np.ndarray,
    N: int,
    user_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_biases: np.ndarray,
    item_vectors: np.ndarray,
    lambd: float,
    gamma: float
) -> None:
    """
    Update item biases in parallel using ALS.

    Formula:
        residual = r_ui - (b_u + u_u . v_i)
        b_i = sum_u lambda * residual / (lambda * |R_i| + gamma)
    """
    K = user_vectors.shape[1]

    for movie_idx in prange(N):
        start = item_map[movie_idx, 0]
        end = item_map[movie_idx, 1]

        if start == end:
            continue

        bias_numerator = 0.0
        for i in range(start, end):
            user_idx = item_users[i]
            rating = item_vals[i]

            dot = 0.0
            for k in range(K):
                dot += user_vectors[user_idx, k] * item_vectors[movie_idx, k]

            prediction_no_bias = user_biases[user_idx] + dot
            residual = rating - prediction_no_bias
            bias_numerator += lambd * residual

        item_biases[movie_idx] = bias_numerator / (lambd * (end - start) + gamma)


@njit(parallel=True)
def update_item_vectors_m_e(
    item_vals: np.ndarray,
    item_users: np.ndarray,
    item_map: np.ndarray,
    N: int,
    user_biases: np.ndarray,
    user_vectors: np.ndarray,
    item_biases: np.ndarray,
    item_vectors: np.ndarray,
    lambd: float,
    tau: float
) -> None:
    """
    Update item latent embeddings in parallel by solving regularized ridge linear systems:
        (lambda * sum_u u_u u_u^T + tau * I) v_i = lambda * sum_u (r_ui - b_u - b_i) u_u
    """
    K = item_vectors.shape[1]

    for movie_idx in prange(N):
        start = item_map[movie_idx, 0]
        end = item_map[movie_idx, 1]

        if start == end:
            continue

        A_matrix = np.zeros((K, K), dtype=np.float64)
        b_vector = np.zeros(K, dtype=np.float64)

        for i in range(start, end):
            user_idx = item_users[i]
            rating = item_vals[i]
            residual = rating - user_biases[user_idx] - item_biases[movie_idx]
            user_vec = user_vectors[user_idx]

            for k1 in range(K):
                for k2 in range(K):
                    A_matrix[k1, k2] += lambd * user_vec[k1] * user_vec[k2]
                b_vector[k1] += lambd * residual * user_vec[k1]

        for k in range(K):
            A_matrix[k, k] += tau

        item_vectors[movie_idx] = np.linalg.solve(A_matrix, b_vector)


def train_model_m_e(
    data_by_user_train: List[List[Tuple[int, float]]],
    data_by_movie_train: List[List[Tuple[int, float]]],
    data_by_user_test: List[List[Tuple[int, float]]],
    M: int,
    N: int,
    K: int = 10,
    lambd: float = 0.1,
    gamma: float = 0.1,
    tau: float = 1.0,
    num_iters: int = 30,
    random_seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train Matrix Factorization model with biases and embeddings using parallel ALS.
    """
    if verbose:
        print("Flattening data structures for ALS...")

    u_vals_tr, u_items_tr, u_map_tr, i_vals_tr, i_users_tr, i_map_tr = prepare_flat_data(
        data_by_user_train, data_by_movie_train, M, N
    )
    u_vals_te, u_items_te, u_map_te, _, _, _ = prepare_flat_data(
        data_by_user_test, [], M, 0
    )

    rng = np.random.default_rng(random_seed)
    user_biases = np.zeros(M, dtype=np.float64)
    item_biases = np.zeros(N, dtype=np.float64)
    user_vectors = rng.normal(0, 0.1, (M, K)).astype(np.float64)
    item_vectors = rng.normal(0, 0.1, (N, K)).astype(np.float64)

    loss_history: List[float] = []
    train_rmse_history: List[float] = []
    test_rmse_history: List[float] = []

    if verbose:
        print("=" * 60)
        print(f"Starting Matrix Factorization ALS (K={K}, lambda={lambd}, gamma={gamma}, tau={tau})")
        print(f"Users: {M:,} | Movies: {N:,} | Latent Dims: {K}")
        print("=" * 60)

    for iteration in range(num_iters):
        # Phase 1: Fix Items, Optimize Users
        update_user_biases_m_e(u_vals_tr, u_items_tr, u_map_tr, M,
                               user_biases, user_vectors, item_biases, item_vectors, lambd, gamma)
        update_user_vectors_m_e(u_vals_tr, u_items_tr, u_map_tr, M,
                                user_biases, user_vectors, item_biases, item_vectors, lambd, tau)

        # Phase 2: Fix Users, Optimize Items
        update_item_biases_m_e(i_vals_tr, i_users_tr, i_map_tr, N,
                               user_biases, user_vectors, item_biases, item_vectors, lambd, gamma)
        update_item_vectors_m_e(i_vals_tr, i_users_tr, i_map_tr, N,
                                user_biases, user_vectors, item_biases, item_vectors, lambd, tau)

        # Evaluation
        train_rmse = evaluate_rmse_m_e(u_vals_tr, u_items_tr, u_map_tr, M,
                                       user_biases, user_vectors, item_biases, item_vectors)
        test_rmse = evaluate_rmse_m_e(u_vals_te, u_items_te, u_map_te, M,
                                      user_biases, user_vectors, item_biases, item_vectors)

        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)

        train_count = len(u_vals_tr)
        train_sse = (train_rmse ** 2) * train_count
        bias_reg = gamma * (np.sum(user_biases ** 2) + np.sum(item_biases ** 2))
        emb_reg = tau * (np.sum(user_vectors ** 2) + np.sum(item_vectors ** 2))
        total_loss = float(train_sse + bias_reg + emb_reg)
        loss_history.append(total_loss)

        if verbose:
            print(f"Iter {iteration + 1:2d}/{num_iters:2d} | Loss: {total_loss:,.0f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")

    if verbose:
        print("=" * 60)
        print("Training Complete!")
        print(f"Final Train RMSE: {train_rmse_history[-1]:.4f}")
        print(f"Final Test RMSE:  {test_rmse_history[-1]:.4f}\n")

    return {
        "user_biases": user_biases,
        "item_biases": item_biases,
        "user_vectors": user_vectors,
        "item_vectors": item_vectors,
        "loss_history": loss_history,
        "train_rmse_history": train_rmse_history,
        "test_rmse_history": test_rmse_history,
    }


def save_mf_model(results: Dict[str, Any], file_path: str = "mf_model.pkl") -> None:
    """Save trained matrix factorization parameters to disk."""
    with open(file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Model saved to {file_path}")


def load_mf_model(file_path: str = "mf_model.pkl") -> Dict[str, Any]:
    """Load saved matrix factorization model parameters from disk."""
    with open(file_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model


class MatrixFactorizationALS:
    """
    Object-oriented Matrix Factorization Recommender.
    """

    def __init__(
        self,
        K: int = 10,
        lambd: float = 0.1,
        gamma: float = 0.1,
        tau: float = 1.0,
        num_iters: int = 30,
        random_seed: int = 42
    ):
        self.K = K
        self.lambd = lambd
        self.gamma = gamma
        self.tau = tau
        self.num_iters = num_iters
        self.random_seed = random_seed
        self.user_biases: Optional[np.ndarray] = None
        self.item_biases: Optional[np.ndarray] = None
        self.user_vectors: Optional[np.ndarray] = None
        self.item_vectors: Optional[np.ndarray] = None
        self.history: Dict[str, List[float]] = {}

    def fit(
        self,
        data_by_user_train: List[List[Tuple[int, float]]],
        data_by_movie_train: List[List[Tuple[int, float]]],
        data_by_user_test: List[List[Tuple[int, float]]],
        M: int,
        N: int,
        verbose: bool = True
    ) -> "MatrixFactorizationALS":
        results = train_model_m_e(
            data_by_user_train=data_by_user_train,
            data_by_movie_train=data_by_movie_train,
            data_by_user_test=data_by_user_test,
            M=M,
            N=N,
            K=self.K,
            lambd=self.lambd,
            gamma=self.gamma,
            tau=self.tau,
            num_iters=self.num_iters,
            random_seed=self.random_seed,
            verbose=verbose
        )
        self.user_biases = results["user_biases"]
        self.item_biases = results["item_biases"]
        self.user_vectors = results["user_vectors"]
        self.item_vectors = results["item_vectors"]
        self.history = {
            "loss": results["loss_history"],
            "train_rmse": results["train_rmse_history"],
            "test_rmse": results["test_rmse_history"],
        }
        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        if self.user_biases is None or self.item_biases is None:
            raise ValueError("Model is not fitted yet.")
        dot = float(np.dot(self.user_vectors[user_idx], self.item_vectors[movie_idx]))
        return float(self.user_biases[user_idx] + self.item_biases[movie_idx] + dot)

    def save(self, file_path: str = "mf_model.pkl") -> None:
        save_mf_model({
            "K": self.K,
            "lambd": self.lambd,
            "gamma": self.gamma,
            "tau": self.tau,
            "user_biases": self.user_biases,
            "item_biases": self.item_biases,
            "user_vectors": self.user_vectors,
            "item_vectors": self.item_vectors,
            "history": self.history
        }, file_path)

    @classmethod
    def load(cls, file_path: str = "mf_model.pkl") -> "MatrixFactorizationALS":
        data = load_mf_model(file_path)
        model = cls(
            K=data.get("K", 10),
            lambd=data.get("lambd", 0.1),
            gamma=data.get("gamma", 0.1),
            tau=data.get("tau", 1.0)
        )
        model.user_biases = data["user_biases"]
        model.item_biases = data["item_biases"]
        model.user_vectors = data["user_vectors"]
        model.item_vectors = data["item_vectors"]
        model.history = data.get("history", {})
        return model
