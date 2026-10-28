"""
CLI script to train Matrix Factorization with Embeddings on MovieLens.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from recsys.data import build_indices, split_train_test
from recsys.models.matrix_factorization import MatrixFactorizationALS


def main():
    parser = argparse.ArgumentParser(description="Train Matrix Factorization Recommender with ALS")
    parser.add_argument("--ratings-path", type=str, default="data/ratings.csv", help="Path to ratings CSV")
    parser.add_argument("--k-dims", type=int, default=10, help="Latent embedding dimensions (K)")
    parser.add_argument("--lambda-param", type=float, default=0.1, help="Rating precision parameter (lambda)")
    parser.add_argument("--gamma-param", type=float, default=0.1, help="Bias L2 regularization parameter (gamma)")
    parser.add_argument("--tau-param", type=float, default=1.0, help="Embedding L2 regularization parameter (tau)")
    parser.add_argument("--iters", type=int, default=20, help="Number of ALS iterations")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--save-model", type=str, default="mf_model.pkl", help="Output path to save model")
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

    model = MatrixFactorizationALS(
        K=args.k_dims,
        lambd=args.lambda_param,
        gamma=args.gamma_param,
        tau=args.tau_param,
        num_iters=args.iters
    )
    model.fit(u_tr, m_tr, u_te, M, N, verbose=True)

    if args.save_model:
        model.save(args.save_model)

    print(f"Matrix Factorization finished. Final Test RMSE: {model.history['test_rmse'][-1]:.4f}")


if __name__ == "__main__":
    main()
