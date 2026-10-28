# Recommender Systems at Scale

[![CI](https://github.com/dev-svald/recommender-system-practicals/actions/workflows/ci.yml/badge.svg)](https://github.com/dev-svald/recommender-system-practicals/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A high-performance, production-grade implementation of **Alternating Least Squares (ALS)** for collaborative filtering and recommender systems. Designed to scale seamlessly from small exploratory datasets (MovieLens 100K) up to massive industrial benchmarks (**MovieLens 32M**: ~330,000 users, ~87,000 items, and 32 million interactions).

Accelerated using **Numba JIT multi-core parallelization** with contiguous flat memory indexing (CSR/CSC layout), delivering up to **10–100× speedups** over native Python.

---

## Table of Contents

- [Highlights](#highlights)
- [Mathematical Formulation](#mathematical-formulation)
  - [Prediction Formula](#prediction-formula)
  - [Objective Loss Function](#objective-loss-function)
  - [ALS Closed-Form Updates](#als-closed-form-updates)
- [Practical Modules](#practical-modules)
  - [Practical 1: Exploratory Data Analysis & Power Law](#practical-1-exploratory-data-analysis--power-law)
  - [Practical 2: Baseline User & Item Bias Model](#practical-2-baseline-user--item-bias-model)
  - [Practical 3: Latent Factor Model with Embeddings](#practical-3-latent-factor-model-with-embeddings)
  - [Practical 4: Going Big on MovieLens 32M](#practical-4-going-big-on-movielens-32m)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Setup](#dataset-setup)
- [Usage](#usage)
  - [1. Running Jupyter Notebooks](#1-running-jupyter-notebooks)
  - [2. Using the CLI Scripts](#2-using-the-cli-scripts)
  - [3. Using the Python API](#3-using-the-python-api)
- [Testing](#testing)
- [License & Acknowledgements](#license--acknowledgements)

---

## Highlights

- **Numba JIT Acceleration**: Parallel loop execution (`@njit(parallel=True)`) solving thread-local linear systems with zero Python GIL overhead.
- **Cache-Friendly Flat Memory Structures**: List-of-lists representations flattened into contiguous 1D NumPy arrays with index pointers, mimicking sparse CSR/CSC memory locality.
- **Complete Recommender Pipeline**: Includes EDA, power-law analysis, baseline biases, embedding matrix factorization, hyperparameter grid searches, cold-start dummy user vector projection, and polarizing movie discovery.
- **Dual Interface**: Provided both as interactive, fully annotated **Jupyter Notebooks** and as a modular, importable **Python package (`recsys`)** with standalone CLI runners.

---

## Mathematical Formulation

### Prediction Formula

For user $u \in \{1, \dots, M\}$ and item $i \in \{1, \dots, N\}$, the predicted rating $\hat{r}_{ui}$ combines global/individual biases with latent factor interaction:

$$\hat{r}_{ui} = b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i$$

where:
- $b_u \in \mathbb{R}$ is the user bias (rating tendency of user $u$).
- $b_i \in \mathbb{R}$ is the item bias (baseline popularity and quality of item $i$).
- $\mathbf{u}_u \in \mathbb{R}^K$ is the latent embedding vector for user $u$.
- $\mathbf{v}_i \in \mathbb{R}^K$ is the latent embedding vector for item $i$.

### Objective Loss Function

The regularized negative log-likelihood objective under Gaussian noise and zero-mean Gaussian priors on parameters is:

$$\mathcal{L} = \frac{1}{2}\lambda \sum_{(u, i) \in \mathcal{R}} \left(r_{ui} - \left(b_u + b_i + \mathbf{u}_u^T \mathbf{v}_i\right)\right)^2 + \frac{1}{2}\gamma \left(\sum_u b_u^2 + \sum_i b_i^2\right) + \frac{1}{2}\tau \left(\sum_u \|\mathbf{u}_u\|_2^2 + \sum_i \|\mathbf{v}_i\|_2^2\right)$$

where $\lambda$ controls data precision, $\gamma$ regularizes biases, and $\tau$ regularizes latent factor magnitudes.

### ALS Closed-Form Updates

Because the objective is quadratic with respect to one set of parameters while fixing the other, Alternating Least Squares (ALS) solves for the global minimum along each coordinate block in closed form:

#### 1. Bias Updates

Fixing embeddings and item biases, the user bias $b_u$ satisfies:

$$b_u = \frac{\lambda \sum_{i \in \mathcal{R}_u} \left(r_{ui} - b_i - \mathbf{u}_u^T \mathbf{v}_i\right)}{\lambda |\mathcal{R}_u| + \gamma}$$

Symmetrically, fixing user parameters, the item bias $b_i$ is updated via:

$$b_i = \frac{\lambda \sum_{u \in \mathcal{R}_i} \left(r_{ui} - b_u - \mathbf{u}_u^T \mathbf{v}_i\right)}{\lambda |\mathcal{R}_i| + \gamma}$$

#### 2. Embedding Vector Updates (Ridge Linear Systems)

Fixing item representations, user embedding $\mathbf{u}_u$ solves a $K \times K$ system:

$$\left(\lambda \sum_{i \in \mathcal{R}_u} \mathbf{v}_i \mathbf{v}_i^T + \tau \mathbf{I}_K\right) \mathbf{u}_u = \lambda \sum_{i \in \mathcal{R}_u} \left(r_{ui} - b_u - b_i\right) \mathbf{v}_i$$

Fixing user representations, item embedding $\mathbf{v}_i$ solves:

$$\left(\lambda \sum_{u \in \mathcal{R}_i} \mathbf{u}_u \mathbf{u}_u^T + \tau \mathbf{I}_K\right) \mathbf{v}_i = \lambda \sum_{u \in \mathcal{R}_i} \left(r_{ui} - b_u - b_i\right) \mathbf{u}_u$$

Each linear system is solved independently and in parallel across available CPU cores via `np.linalg.solve`.

---

## Practical Modules

### Practical 1: Exploratory Data Analysis & Power Law
- **Rating Sparsity**: Analysis of user and movie activity distribution.
- **Power Law**: Fitting log-log distributions showing that a small fraction of "blockbuster" movies capture the vast majority of interactions.
- **Index Construction**: Conversion of arbitrary string or sparse numeric IDs into zero-based contiguous indexes.

### Practical 2: Baseline User & Item Bias Model
- Alternating updates for $b_u$ and $b_i$ without latent factors.
- Fast evaluation of Root Mean Squared Error (RMSE) on disjoint train/test partitions.
- Grid search over precision $\lambda$ and regularization $\gamma$ with heatmap visualizations.

### Practical 3: Latent Factor Model with Embeddings
- Full Matrix Factorization incorporating $K$-dimensional vectors.
- Multi-core parallelization benchmarks comparing native Python against Numba JIT.
- Staged hyperparameter search exploring latent dimension $K \in \{2, 5, 10, 20\}$, bias penalty $\gamma$, and embedding penalty $\tau$.

### Practical 4: Going Big on MovieLens 32M
- Scaled training on the full 32-million interaction MovieLens dataset.
- **Dummy User Projection**: Cold-start inference projecting liked movie vectors into latent space to generate personalized Top-$K$ recommendations.
- **Polarizing vs. Consensus Movies**: Geometric analysis of embedding norms $\|\mathbf{v}_i\|_2$ to identify polarizing cult classics versus universally agreed consensus movies.
- **2D Latent Geometry**: Training a 2D model to project and interpret the semantic space of genres and movie affinities.

---

## Repository Structure

```text
recommender-system-practicals/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (Python 3.10-3.12)
├── data/                        # Local dataset storage (ignored by git)
│   └── .gitkeep
├── notebooks/                   # Interactive Jupyter practicals
│   ├── 01_ml_at_scale_intro.ipynb
│   └── 02_ml_at_scale_final.ipynb
├── scripts/                     # Standalone CLI tools
│   ├── train_baseline.py        # Train baseline bias model
│   ├── train_mf.py              # Train matrix factorization with ALS
│   └── interactive_recommend.py # CLI for personalized recommendations
├── src/
│   └── recsys/                  # Modular Python library
│       ├── __init__.py
│       ├── data.py              # Indexing, splitting, flat CSR memory flattening
│       ├── evaluation.py        # RMSE, SSE, and loss tracking (Numba + NumPy)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline.py      # Baseline user/item bias ALS model
│       │   └── matrix_factorization.py  # Matrix Factorization ALS model
│       ├── recommend.py         # Dummy user vector projection & Top-K ranking
│       ├── tuning.py            # Grid searches and parameter sweep utilities
│       └── visualization.py     # Training curves, heatmaps, 2D latent projections
├── tests/                       # Unit test suite
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
├── .gitignore
├── LICENSE                      # MIT License
├── pyproject.toml               # Modern build configuration & dependency spec
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.9+ (Python 3.10–3.12 recommended)
- Git

### Installation

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/dev-svald/recommender-system-practicals.git
cd recommender-system-practicals

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

### Dataset Setup

Download the MovieLens dataset (MovieLens 100K, 1M, or 32M) and extract it into `data/`:

```bash
# Example for MovieLens 32M:
curl -O https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip -d data/
```

The expected file structure in `data/`:
```text
data/
├── movies.csv
└── ratings.csv
```

---

## Usage

### 1. Running Jupyter Notebooks

Launch JupyterLab to interact with the full step-by-step practicals:

```bash
jupyter lab notebooks/
```

- **`01_ml_at_scale_intro.ipynb`**: Foundation practicals, EDA, power law, baseline bias model.
- **`02_ml_at_scale_final.ipynb`**: Comprehensive practicals, Numba parallelization, MovieLens 32M scaling, recommendations, and geometric visualizations.

### 2. Using the CLI Scripts

#### Train Baseline Bias Model:
```bash
python scripts/train_baseline.py --ratings-path data/ratings.csv --iters 20 --lambda-param 0.1 --gamma-param 0.1
```

#### Train Matrix Factorization with ALS:
```bash
python scripts/train_mf.py --ratings-path data/ratings.csv --k-dims 10 --iters 20 --save-model mf_model.pkl
```

#### Generate Personalized Recommendations:
```bash
python scripts/interactive_recommend.py \
    --model-path mf_model.pkl \
    --movies-csv data/movies.csv \
    --movies "Toy Story" "Monsters, Inc." "Finding Nemo" \
    --bias-weight 0.05 \
    --top-k 10
```

### 3. Using the Python API

```python
from recsys.data import build_indices, split_train_test
from recsys.models import MatrixFactorizationALS
from recsys.recommend import create_dummy_user, recommend_movies

# 1. Index and split interactions
(u_map, idx_to_u, data_u), (m_map, idx_to_m, data_m) = build_indices(ratings)
u_tr, u_te, m_tr, m_te = split_train_test(data_u, len(idx_to_u), len(idx_to_m))

# 2. Fit Matrix Factorization ALS model
model = MatrixFactorizationALS(K=10, lambd=0.1, gamma=0.1, tau=1.0, num_iters=20)
model.fit(u_tr, m_tr, u_te, len(idx_to_u), len(idx_to_m))

# 3. Save model weights
model.save("mf_model.pkl")
```

---

## Testing

Run the automated test suite:

```bash
# Using Python's built-in unittest runner:
python -m unittest discover -s tests

# Or using pytest:
pytest tests/
```

---

## License & Acknowledgements

- **License**: Released under the [MIT License](LICENSE).
- **Author**: [`dev-svald`](https://github.com/dev-svald).
- **Upstream Source**: Inspired by and ported from [`naoussijames/Recommendetr-system-practicals`](https://github.com/naoussijames/Recommendetr-system-practicals).
- **Dataset**: MovieLens datasets provided by [GroupLens Research](https://grouplens.org/datasets/movielens/).
