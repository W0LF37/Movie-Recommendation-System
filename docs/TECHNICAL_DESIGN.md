# Technical Design — Original Notebook

This document explains the original academic implementation without claiming a modular library, API, service, or new engineering work.

## 1. Inputs and preparation

- The notebook downloads the MovieLens 20M archive and loads `ratings.csv` and `movies.csv`.
- Movie metadata is joined with ratings; movie identifiers are mapped to a contiguous ID range.
- UNIX timestamps are converted to date strings; records are sorted chronologically.
- The first 80% of sorted rating rows form training data; the remainder forms test data, cached as pandas pickle files.

## 2. Representation and exploratory analysis

- SciPy CSR matrices encode explicit user–movie ratings.
- The notebook inspects matrix sparsity, ratings, user/movie activity and date trends.
- Cosine similarity is used for movie–movie retrieval and user–user similarity.

## 3. Feature engineering

The notebook samples sparse matrices in its later modeling stage to manage compute cost. Each user–movie observation yields 13 predictor variables:

| Feature group | Count | Meaning |
| --- | ---: | --- |
| Global average | 1 | Mean observed training-sample rating |
| User average | 1 | User's average rating |
| Movie average | 1 | Movie's average rating |
| SUR1–SUR5 | 5 | Ratings by similar users for the target movie |
| SMR1–SMR5 | 5 | User's ratings for similar movies |
| **Total** | **13** | Prediction context excluding IDs and rating target |

Missing-neighbor contexts are filled with average-rating fallback values. The features are assembled as data frames for regression.

## 4. Model comparison

- `XGBRegressor` is evaluated first on the 13 handcrafted features.
- Surprise's `BaselineOnly`, `KNNBaseline` (user-user and item-item), `SlopeOne`, `SVD`, and `SVDpp` estimate ratings.
- Additional XGBoost configurations consume predicted values from Surprise models as extra features.
- The notebook runs parameter searches for KNNBaseline, SVD, and SVDpp, using 3-fold cross-validation for shown Surprise searches.
- RMSE and MAPE are calculated and recorded for train and test predictions.

## 5. Recommendation generation

SVD++ is separately trained on the sample training set. The notebook constructs an anti-testset of unseen user–item pairs, predicts their ratings, sorts them descending per user and returns a Top-N list (default 10) joined with movie titles and genres.

**Important distinction:** The lowest saved test-RMSE configuration is an XGBoost ensemble, while the demonstrated Top-N recommendations come from SVD++. These are different notebook outputs.

## 6. Execution characteristics

- Original environment: notebook/Google Colab with `/content/ml-20m` absolute paths.
- Intermediate `.pkl`, `.npz`, and `.csv` artifacts are stored under the data folder.
- The notebook does not implement an application, deployment endpoint, model registry, automated tests, or scheduled training pipeline.
