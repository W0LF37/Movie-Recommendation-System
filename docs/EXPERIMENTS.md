# Experiments, Recorded Results, and Limitations

## What the saved notebook shows

The notebook contains 11 recorded train/test RMSE comparisons. Values below are copied from the final `error_table` saved output (rounded to four decimals).

| Model | Train RMSE | Test RMSE |
| --- | ---: | ---: |
| XGBoost_13 | 0.8072 | 1.0048 |
| BaselineOnly | 0.8274 | 0.9996 |
| XGB_BSL | 0.8073 | 1.0036 |
| KNNBaseline_User | 0.3378 | 0.9995 |
| KNNBaseline_Item | 0.2742 | 0.9995 |
| XGB_BSL_KNN | 0.8074 | 1.0042 |
| SlopeOne | 0.7606 | 0.9991 |
| SVD | 0.8311 | 0.9995 |
| SVDpp | 0.7631 | 0.9995 |
| XGB_BSL_KNN_MF | 0.8071 | 1.0052 |
| XGB_KNN_MF_SO | 1.0665 | **0.9981** |

The test-RMSE differences between configurations are small. These figures should not be interpreted as proof of superiority on other datasets or offline ranking metrics.

## Documented hyperparameter searches

- SVD: searched `n_factors` values; saved best 3-fold CV RMSE **0.9023683** with **5 factors**.
- SVD++: searched `n_factors` and `lr_all`; saved best 3-fold CV RMSE **0.8964136** with **10 factors**, **learning rate 0.006**.
- Other model configurations and parameter selections are defined in the original notebook.

**Do not compare** these CV scores directly to the final test values as if they came from identical evaluation procedures.

## Methodological interpretation

1. **Split:** The data is sorted by date and split into an 80/20 rating-row partition. It is not a per-user holdout protocol. The original notebook reports users missing from the training portion.
2. **Sampling:** Training and testing use sampled sparse interaction matrices for feature/model stages; the experiment is not a claim that all 20 million ratings were used for every fitted model.
3. **Stacking / leakage risk:** Some Surprise models' predictions for training rows are obtained from models fitted on those same training rows and then used as XGBoost training features. An out-of-fold stacking procedure is not shown. Consequently, ensemble metrics should be interpreted cautiously.
4. **Potential target contamination:** Some handcrafted rating-context statistics are computed from the sampled training interaction matrix containing the training targets, so their train error estimates may be optimistic.
5. **Metrics:** RMSE and MAPE assess predicted explicit ratings, not Top-N recommendation relevance. Ranking-specific metrics were not reported.
6. **Reproducibility:** Saved cell outputs are historical records, not a verified re-run in a pinned environment. The notebook contains older APIs, e.g. pandas `DataFrame.append`, that may fail in updated environments.

## Responsible summary for a portfolio

> Built an academic movie recommendation study on MovieLens 20M, exploring sparse user–item interactions, similarity-based collaborative filtering, neighborhood-derived features, matrix factorization and gradient-boosted rating regression. Compared 11 model configurations and generated personalized Top-10 movie lists using SVD++. The saved experiment's lowest test RMSE was 0.9981, subject to the documented evaluation limitations.
