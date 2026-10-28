"""
Hyperparameter tuning and grid search utilities for baseline and embedding models.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

from recsys.models.baseline import train_bias_model


def grid_search_bias_hyperparameters(
    data_by_user_train: List[List],
    data_by_movie_train: List[List],
    data_by_user_test: List[List],
    M: int,
    N: int,
    lambda_values: List[float],
    gamma_values: List[float],
    iter_values: List[int],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Exhaustive grid search across bias hyperparameters (lambda, gamma, num_iters).
    """
    records = []
    best_test_rmse = float("inf")
    best_config: Optional[Dict[str, Any]] = None
    best_model_results: Optional[Dict[str, Any]] = None

    total_configs = len(lambda_values) * len(gamma_values) * len(iter_values)
    idx = 0

    for lambd, gamma, iters in itertools.product(lambda_values, gamma_values, iter_values):
        idx += 1
        if verbose:
            print(f"[{idx}/{total_configs}] Testing: lambda={lambd}, gamma={gamma}, iters={iters}...")

        res = train_bias_model(
            data_by_user_train=data_by_user_train,
            data_by_movie_train=data_by_movie_train,
            data_by_user_test=data_by_user_test,
            M=M,
            N=N,
            lambd=lambd,
            gamma=gamma,
            num_iters=iters,
            verbose=False
        )

        train_rmse = res["train_rmse_history"][-1]
        test_rmse = res["test_rmse_history"][-1]
        gap = test_rmse - train_rmse

        records.append({
            "lambda": lambd,
            "gamma": gamma,
            "num_iters": iters,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "gap": gap
        })

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_config = {"lambda": lambd, "gamma": gamma, "num_iters": iters}
            best_model_results = res

    results_df = pd.DataFrame(records).sort_values("test_rmse").reset_index(drop=True)
    return {
        "results": results_df,
        "best_config": best_config,
        "best_model": best_model_results
    }
