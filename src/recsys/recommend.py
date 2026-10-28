"""
Inference, personalized recommendation generation, and polarizing items analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def create_dummy_user(
    loved_movie_titles: List[str],
    user_vectors: np.ndarray,
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    idx_to_movieid: List[int],
    movie_id_to_title: Dict[int, str],
    K: int = 10
) -> Tuple[np.ndarray, List[int]]:
    """
    Construct a synthetic user embedding by matching loved movie queries
    and projecting the average of their learned latent representations.

    Parameters
    ----------
    loved_movie_titles : list of str
        Target titles to match against dataset.
    user_vectors : np.ndarray (M, K)
    item_vectors : np.ndarray (N, K)
    item_biases : np.ndarray (N,)
    idx_to_movieid : list
        Mapping from internal index to original movie ID.
    movie_id_to_title : dict
        Mapping from movie ID to title string.
    K : int, default=10

    Returns
    -------
    dummy_user_vector : np.ndarray (K,)
        Synthetic user representation in latent space.
    loved_movie_indices : list of int
        Internal indices of successfully matched movies.
    """
    loved_movie_indices: List[int] = []

    for query in loved_movie_titles:
        query_clean = query.strip().lower()
        found = False
        for idx, movie_id in enumerate(idx_to_movieid):
            title = movie_id_to_title.get(movie_id, "Unknown")
            if query_clean in title.lower():
                loved_movie_indices.append(idx)
                found = True
                break
        if not found:
            print(f"Warning: Loved movie '{query}' not found in dataset.")

    if not loved_movie_indices:
        raise ValueError("None of the specified loved movies were found in the dataset.")

    loved_vectors = item_vectors[loved_movie_indices]
    dummy_user_vector = np.mean(loved_vectors, axis=0)
    return dummy_user_vector, loved_movie_indices


def recommend_movies(
    dummy_user_vector: np.ndarray,
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    idx_to_movieid: List[int],
    movie_id_to_title: Dict[int, str],
    bias_weight: float = 1.0,
    min_ratings: int = 0,
    movie_rating_counts: Optional[Dict[int, int]] = None,
    exclude_indices: Optional[List[int]] = None,
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Generate Top-K recommendations ranked by combined personalization and popularity score.

    Score formulation:
        score = (item_vectors @ dummy_user_vector) + bias_weight * item_biases

    Parameters
    ----------
    dummy_user_vector : np.ndarray (K,)
    item_vectors : np.ndarray (N, K)
    item_biases : np.ndarray (N,)
    idx_to_movieid : list
    movie_id_to_title : dict
    bias_weight : float, default=1.0
        Weight given to overall item popularity (bias).
        - 1.0: standard rating prediction
        - 0.05: downweight popularity in favor of pure personalization
        - 0.0: cosine/dot-product alignment only
    min_ratings : int, default=0
        Minimum rating threshold to avoid recommending obscure edge-case items.
    movie_rating_counts : dict, optional
    exclude_indices : list of int, optional
        Indices already consumed (e.g. loved movies).
    top_k : int, default=20

    Returns
    -------
    list of dicts containing recommendation records.
    """
    N = len(item_vectors)
    dot_products = item_vectors @ dummy_user_vector
    bias_contribution = bias_weight * item_biases
    scores = dot_products + bias_contribution

    exclude_set = set(exclude_indices) if exclude_indices else set()
    results: List[Dict[str, Any]] = []

    for idx in range(N):
        if idx in exclude_set:
            continue

        count = movie_rating_counts.get(idx, 0) if movie_rating_counts else 0
        if count < min_ratings:
            continue

        movie_id = idx_to_movieid[idx]
        title = movie_id_to_title.get(movie_id, "Unknown")

        results.append({
            "movie_idx": idx,
            "score": float(scores[idx]),
            "dot_product": float(dot_products[idx]),
            "bias_contribution": float(bias_contribution[idx]),
            "title": title,
            "rating_count": count
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def display_recommendations(
    recommendations: List[Dict[str, Any]],
    loved_movie_titles: Optional[List[str]] = None,
    bias_weight: float = 1.0
) -> None:
    """Pretty-print recommendations with score breakdown."""
    print("=" * 95)
    print(f"TOP {len(recommendations)} RECOMMENDATIONS (Bias weight: {bias_weight})")
    print("=" * 95)

    if loved_movie_titles:
        print(f"Seed preferences: {', '.join(loved_movie_titles)}")

    header = f"{'Rank':<6} {'Score':<8} {'Personal':<10} {'Popularity':<10} {'Ratings':<8} {'Title':<50}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, rec in enumerate(recommendations, 1):
        print(
            f"{i:<6} {rec['score']:<8.3f} {rec['dot_product']:<10.3f} "
            f"{rec['bias_contribution']:<10.3f} {rec['rating_count']:<8,} "
            f"{rec['title'][:48]:<50}"
        )
    print("=" * 95)


def find_polarizing_movies(
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    idx_to_movieid: List[int],
    movie_id_to_title: Dict[int, str],
    movie_rating_counts: Dict[int, int],
    min_ratings: int = 100,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Identify polarizing movies using embedding vector magnitude ||v_i||.
    Movies with high embedding norm push users toward extreme positive or negative predictions.
    """
    N = len(item_vectors)
    vector_norms = np.linalg.norm(item_vectors, axis=1)

    results = []
    for idx in range(N):
        count = movie_rating_counts.get(idx, 0)
        if count < min_ratings:
            continue

        movie_id = idx_to_movieid[idx]
        title = movie_id_to_title.get(movie_id, "Unknown")
        results.append({
            "movie_idx": idx,
            "title": title,
            "vector_norm": float(vector_norms[idx]),
            "bias": float(item_biases[idx]),
            "rating_count": count,
            "movie_id": movie_id
        })

    df = pd.DataFrame(results)
    return df.sort_values("vector_norm", ascending=False).reset_index(drop=True).head(top_k)


def find_non_polarizing_movies(
    item_vectors: np.ndarray,
    item_biases: np.ndarray,
    idx_to_movieid: List[int],
    movie_id_to_title: Dict[int, str],
    movie_rating_counts: Dict[int, int],
    min_ratings: int = 100,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Identify consensus (non-polarizing) movies characterized by small embedding norms.
    """
    N = len(item_vectors)
    vector_norms = np.linalg.norm(item_vectors, axis=1)

    results = []
    for idx in range(N):
        count = movie_rating_counts.get(idx, 0)
        if count < min_ratings:
            continue

        movie_id = idx_to_movieid[idx]
        title = movie_id_to_title.get(movie_id, "Unknown")
        results.append({
            "movie_idx": idx,
            "title": title,
            "vector_norm": float(vector_norms[idx]),
            "bias": float(item_biases[idx]),
            "rating_count": count,
            "movie_id": movie_id
        })

    df = pd.DataFrame(results)
    return df.sort_values("vector_norm", ascending=True).reset_index(drop=True).head(top_k)
