"""
CLI script to generate personalized movie recommendations using trained model embeddings.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from recsys.data import load_movie_metadata
from recsys.models.matrix_factorization import MatrixFactorizationALS
from recsys.recommend import create_dummy_user, display_recommendations, recommend_movies


def main():
    parser = argparse.ArgumentParser(description="Generate Movie Recommendations for Seed Movies")
    parser.add_argument("--model-path", type=str, default="mf_model.pkl", help="Path to saved MF model pickle")
    parser.add_argument("--movies-csv", type=str, default="data/movies.csv", help="Path to movies.csv")
    parser.add_argument("--movies", nargs="+", default=["Toy Story", "Star Wars"], help="Movie titles liked by user")
    parser.add_argument("--bias-weight", type=float, default=0.05, help="Popularity bias weight (0.0=pure personalization)")
    parser.add_argument("--top-k", type=int, default=15, help="Number of movies to recommend")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Trained model '{args.model_path}' not found. Run scripts/train_mf.py first.")
        sys.exit(1)
    if not os.path.exists(args.movies_csv):
        print(f"Error: Movies metadata '{args.movies_csv}' not found.")
        sys.exit(1)

    print(f"Loading model from {args.model_path}...")
    model = MatrixFactorizationALS.load(args.model_path)

    print(f"Loading metadata from {args.movies_csv}...")
    movies_df, movie_id_to_title = load_movie_metadata(args.movies_csv)
    idx_to_movieid = list(movies_df["movieId"])

    dummy_vec, loved_indices = create_dummy_user(
        loved_movie_titles=args.movies,
        user_vectors=model.user_vectors,
        item_vectors=model.item_vectors,
        item_biases=model.item_biases,
        idx_to_movieid=idx_to_movieid,
        movie_id_to_title=movie_id_to_title,
        K=model.K
    )

    recommendations = recommend_movies(
        dummy_user_vector=dummy_vec,
        item_vectors=model.item_vectors,
        item_biases=model.item_biases,
        idx_to_movieid=idx_to_movieid,
        movie_id_to_title=movie_id_to_title,
        bias_weight=args.bias_weight,
        exclude_indices=loved_indices,
        top_k=args.top_k
    )

    display_recommendations(recommendations, loved_movie_titles=args.movies, bias_weight=args.bias_weight)


if __name__ == "__main__":
    main()
