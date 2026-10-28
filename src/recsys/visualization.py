"""
Visualization utilities for model evaluation, hyperparameter searches, and embedding analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


def _check_plotting_deps() -> None:
    if not HAS_PLOTTING:
        raise ImportError(
            "Visualization functions require 'matplotlib' and 'seaborn'. "
            "Install them with: pip install matplotlib seaborn"
        )


def plot_training_curves(
    model_results: Dict[str, Any],
    title: str = "ALS Optimization Progress",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training loss and RMSE curves across ALS iterations.
    """
    _check_plotting_deps()

    loss_history = model_results["loss_history"]
    train_rmse_history = model_results["train_rmse_history"]
    test_rmse_history = model_results["test_rmse_history"]
    num_iters = len(train_rmse_history)
    iters = range(1, num_iters + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(iters, loss_history, "b-o", linewidth=2, markersize=4)
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Loss / SSE", fontsize=11)
    ax1.set_title("Training Loss across Iterations", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # RMSE curves
    ax2.plot(iters, train_rmse_history, "g-o", label="Train RMSE", linewidth=2, markersize=4)
    ax2.plot(iters, test_rmse_history, "r--s", label="Test RMSE", linewidth=2, markersize=4)
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("RMSE", fontsize=11)
    ax2.set_title("Generalization Performance (Train vs Test RMSE)", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training plot to {save_path}")
    plt.close()


def plot_hyperparameter_heatmaps(
    results_df: pd.DataFrame,
    save_path: Optional[str] = "hyperparameter_heatmaps.pdf"
) -> None:
    """
    Generate test RMSE heatmaps for hyperparameter combinations.
    """
    _check_plotting_deps()

    iter_values = sorted(results_df["num_iters"].unique())
    n_configs = len(iter_values)

    fig, axes = plt.subplots(1, n_configs, figsize=(6 * n_configs, 5), squeeze=False)

    for idx, iters in enumerate(iter_values):
        sub = results_df[results_df["num_iters"] == iters]
        pivot = sub.pivot(index="gamma", columns="lambda", values="test_rmse")

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".4f",
            cmap="viridis_r",
            cbar=True,
            ax=axes[0, idx]
        )
        axes[0, idx].set_title(f"Test RMSE (Iters = {iters})", fontsize=12)
        axes[0, idx].set_xlabel(r"$\lambda$", fontsize=11)
        axes[0, idx].set_ylabel(r"$\gamma$", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved heatmaps to {save_path}")
    plt.close()


def plot_2d_embeddings(
    selected_movies_df: pd.DataFrame,
    item_vectors_2d: np.ndarray,
    save_path: Optional[str] = "embedding_space_2d.pdf"
) -> None:
    """
    Plot 2-dimensional movie embeddings in latent space.
    """
    _check_plotting_deps()

    plt.figure(figsize=(12, 10))
    x = item_vectors_2d[:, 0]
    y = item_vectors_2d[:, 1]

    plt.scatter(x, y, alpha=0.7, c="royalblue", edgecolors="black", s=60)

    for _, row in selected_movies_df.iterrows():
        idx = int(row["movie_idx"])
        title = row["title"]
        plt.annotate(
            title,
            (item_vectors_2d[idx, 0], item_vectors_2d[idx, 1]),
            fontsize=8,
            alpha=0.85
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Latent Dimension 1", fontsize=11)
    plt.ylabel("Latent Dimension 2", fontsize=11)
    plt.title("2D Movie Embedding Space Geometry", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved 2D embedding plot to {save_path}")
    plt.close()
