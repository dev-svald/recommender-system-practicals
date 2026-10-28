"""
Data loading, preprocessing, and indexing utilities for recommender systems.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def build_indices(
    ratings: List[Tuple[int, int, float]]
) -> Tuple[
    Tuple[Dict[int, int], List[int], List[List[Tuple[int, float]]]],
    Tuple[Dict[int, int], List[int], List[List[Tuple[int, float]]]]
]:
    """
    Build index structures mapping original user and movie IDs to contiguous 0-based indices.

    Parameters
    ----------
    ratings : list of (user_id, movie_id, rating)
        Raw ratings list.

    Returns
    -------
    user_structures : tuple
        (userid_to_idx, idx_to_userid, data_by_user)
    movie_structures : tuple
        (movieid_to_idx, idx_to_movieid, data_by_movie)
    """
    userid_to_idx: Dict[int, int] = {}
    idx_to_userid: List[int] = []
    data_by_user: List[List[Tuple[int, float]]] = []

    movieid_to_idx: Dict[int, int] = {}
    idx_to_movieid: List[int] = []
    data_by_movie: List[List[Tuple[int, float]]] = []

    for user_id, movie_id, rating in ratings:
        user_id = int(user_id)
        movie_id = int(movie_id)
        r = float(rating)

        if user_id not in userid_to_idx:
            userid_to_idx[user_id] = len(idx_to_userid)
            idx_to_userid.append(user_id)
            data_by_user.append([])

        if movie_id not in movieid_to_idx:
            movieid_to_idx[movie_id] = len(idx_to_movieid)
            idx_to_movieid.append(movie_id)
            data_by_movie.append([])

        u_idx = userid_to_idx[user_id]
        m_idx = movieid_to_idx[movie_id]

        data_by_user[u_idx].append((m_idx, r))
        data_by_movie[m_idx].append((u_idx, r))

    return (
        (userid_to_idx, idx_to_userid, data_by_user),
        (movieid_to_idx, idx_to_movieid, data_by_movie)
    )


def split_train_test(
    data_by_user: List[List[Tuple[int, float]]],
    number_of_users: int,
    number_of_movies: int,
    train_ratio: float = 0.9,
    random_seed: int = 42,
    verbose: bool = True
) -> Tuple[
    List[List[Tuple[int, float]]],
    List[List[Tuple[int, float]]],
    List[List[Tuple[int, float]]],
    List[List[Tuple[int, float]]]
]:
    """
    Split per-user rating data into reproducible training and testing sets.

    Parameters
    ----------
    data_by_user : list of list of (movie_idx, rating)
        Ratings grouped by internal user index.
    number_of_users : int
        Total number of users.
    number_of_movies : int
        Total number of movies.
    train_ratio : float, default=0.9
        Proportion of ratings per user allocated to training.
    random_seed : int, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print partition statistics.

    Returns
    -------
    tuple of (data_by_user_train, data_by_user_test,
              data_by_movie_train, data_by_movie_test)
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if len(data_by_user) != number_of_users:
        raise ValueError(f"Length of data_by_user ({len(data_by_user)}) != number_of_users ({number_of_users})")

    rng = random.Random(random_seed)

    data_by_user_train: List[List[Tuple[int, float]]] = [[] for _ in range(number_of_users)]
    data_by_user_test: List[List[Tuple[int, float]]] = [[] for _ in range(number_of_users)]
    data_by_movie_train: List[List[Tuple[int, float]]] = [[] for _ in range(number_of_movies)]
    data_by_movie_test: List[List[Tuple[int, float]]] = [[] for _ in range(number_of_movies)]

    for user_idx in range(number_of_users):
        user_ratings = list(data_by_user[user_idx])
        rng.shuffle(user_ratings)

        split_point = int(train_ratio * len(user_ratings))
        train_ratings = user_ratings[:split_point]
        test_ratings = user_ratings[split_point:]

        data_by_user_train[user_idx] = train_ratings
        data_by_user_test[user_idx] = test_ratings

        for movie_idx, rating in train_ratings:
            data_by_movie_train[movie_idx].append((user_idx, rating))
        for movie_idx, rating in test_ratings:
            data_by_movie_test[movie_idx].append((user_idx, rating))

    if verbose:
        total_ratings = sum(len(r) for r in data_by_user)
        train_count = sum(len(r) for r in data_by_user_train)
        test_count = sum(len(r) for r in data_by_user_test)
        print(f"Total ratings: {total_ratings:,}")
        print(f"Train ratings: {train_count:,} ({train_count/total_ratings*100:.1f}%)")
        print(f"Test ratings:  {test_count:,} ({test_count/total_ratings*100:.1f}%)")

    return (
        data_by_user_train, data_by_user_test,
        data_by_movie_train, data_by_movie_test
    )


def prepare_flat_data(
    data_by_user: List[List[Tuple[int, float]]],
    data_by_movie: List[List[Tuple[int, float]]],
    M: int,
    N: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten list-of-lists into contiguous 1D NumPy arrays (CSR/CSC layout)
    for high-speed SIMD and Numba parallel loop processing.

    Returns
    -------
    user_vals : np.ndarray (float64)
        Rating values ordered by user.
    user_idx : np.ndarray (int32)
        Movie indices ordered by user.
    user_map : np.ndarray (int32, shape=(M, 2))
        Pointer array where row u has [start_index, end_index].
    item_vals : np.ndarray (float64)
        Rating values ordered by movie.
    item_idx : np.ndarray (int32)
        User indices ordered by movie.
    item_map : np.ndarray (int32, shape=(N, 2))
        Pointer array where row i has [start_index, end_index].
    """
    # Flatten user data
    total_user_ratings = sum(len(u) for u in data_by_user) if data_by_user else 0
    user_vals = np.zeros(total_user_ratings, dtype=np.float64)
    user_idx = np.zeros(total_user_ratings, dtype=np.int32)
    user_map = np.zeros((M, 2), dtype=np.int32)

    current_ptr = 0
    for u in range(min(M, len(data_by_user))):
        ratings = data_by_user[u]
        count = len(ratings)
        user_map[u, 0] = current_ptr
        user_map[u, 1] = current_ptr + count
        for i, (item_id, rating) in enumerate(ratings):
            user_idx[current_ptr + i] = item_id
            user_vals[current_ptr + i] = rating
        current_ptr += count

    # Flatten item data
    total_item_ratings = sum(len(i) for i in data_by_movie) if data_by_movie else 0
    item_vals = np.zeros(total_item_ratings, dtype=np.float64)
    item_idx = np.zeros(total_item_ratings, dtype=np.int32)
    item_map = np.zeros((N, 2), dtype=np.int32)

    current_ptr = 0
    for i in range(min(N, len(data_by_movie))):
        ratings = data_by_movie[i]
        count = len(ratings)
        item_map[i, 0] = current_ptr
        item_map[i, 1] = current_ptr + count
        for j, (user_id, rating) in enumerate(ratings):
            item_idx[current_ptr + j] = user_id
            item_vals[current_ptr + j] = rating
        current_ptr += count

    return user_vals, user_idx, user_map, item_vals, item_idx, item_map


def load_movie_metadata(movies_csv_path: str = "ml-32m/movies.csv") -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Load movie titles, IDs, and genres from MovieLens metadata.

    Parameters
    ----------
    movies_csv_path : str
        Path to movies.csv.

    Returns
    -------
    movies_df : pd.DataFrame
        DataFrame with columns [movieId, title, genres].
    movie_id_to_title : dict
        Mapping from original movieId to title string.
    """
    movies_df = pd.read_csv(movies_csv_path)
    movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    return movies_df, movie_id_to_title


def compute_movie_rating_counts(data_by_movie: List[List[Tuple[int, float]]]) -> Dict[int, int]:
    """
    Compute rating counts per movie index.
    """
    return {movie_idx: len(ratings) for movie_idx, ratings in enumerate(data_by_movie)}
