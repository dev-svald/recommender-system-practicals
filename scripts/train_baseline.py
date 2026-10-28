"""
CLI script to train the Baseline User/Item Bias Model on MovieLens datasets.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from recsys.data import build_indices, split_train_test
from recsys.models.baseline import BaselineBiasModel


def main():
    parser = argparse.ArgumentParser(description="Train Baseline User/Item Bias Recommender")
    parser.add_argument("--ratings-path", type=str, default="data/ratings.csv", help="Path to ratings CSV")
    parser.add_argument("--lambda-param", type=float, default=0.1, help="Rating precision parameter (lambda)")
    parser.add_argument("--gamma-param", type=float, default=0.1, help="Bias L2 regularization parameter (gamma)")
    parser.add_argument("--iters", type=int, default=20, help="Number of ALS iterations")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    args = parser.parse_args()

    if not os.path.exists(args.ratings_path):
        print(f"Error: Ratings file not found at '{args.ratings_path}'.")
        print("Please download MovieLens dataset into data/ or provide --ratings-path.")
        sys.exit(1)

    print(f"Loading ratings from {args.ratings_path}...")
    df = pd.read_csv(args.ratings_path)
    ratings = list(zip(df["userId"], df["movieId"], df["rating"]))

    (u_map, idx_to_u, data_u), (m_map, idx_to_m, data_m) = build_indices(ratings)
    M = len(idx_to_u)
    N = len(idx_to_m)

    u_tr, u_te, m_tr, m_te = split_train_test(
        data_u, number_of_users=M, number_of_movies=N, train_ratio=args.train_ratio
    )

    model = BaselineBiasModel(lambd=args.lambda_param, gamma=args.gamma_param, num_iters=args.iters)
    model.fit(u_tr, m_tr, u_te, M, N, verbose=True)

    print(f"Baseline training finished. Final Test RMSE: {model.history['test_rmse'][-1]:.4f}")


if __name__ == "__main__":
    main()
