# Tree Models · Quick Reference

**Random Forest · XGBoost · LightGBM**

All three support **regression and classification**. Ranges below are suggested starting points, not fixed limits or library defaults.

## At a glance

| Model | How it works |
| :--- | :--- |
| **Random Forest** | Trains separate trees and averages their predictions. |
| **XGBoost** | Adds trees sequentially to improve predictions. |
| **LightGBM** | Also uses boosting; grows the most useful leaf first and uses histogram bins for speed. |

## XGBoost

| Parameter | Controls | Starting range / value |
| :--- | :--- | :--- |
| `n_estimators` | Boosting rounds | **300–2,000**, with early stopping |
| `learning_rate` | Each tree's contribution | **0.01–0.2** |
| `max_depth` | Tree depth | **3–8** |
| `min_child_weight` | Minimum Hessian per leaf | **1–10** |
| `subsample` | Fraction of rows per round | **0.6–1.0** |
| `colsample_bytree` | Fraction of features per tree | **0.6–1.0** |
| `reg_alpha` | L1 penalty on leaf outputs | **0, 0.01, 0.1, 1, 10** |
| `reg_lambda` | L2 penalty on leaf outputs | **0.1, 1, 10** |
| `gamma` | Minimum gain required to split | **0** initially; scale-dependent |
| `early_stopping_rounds` | Rounds without validation improvement | **50–100** |
| `max_bin` | Histogram resolution | **128 or 256**, with `tree_method="hist"` |
| `max_leaves` | Maximum leaves | Usually **0** (unlimited); optional with histogram trees |

**Simple start:** depth **4**, learning rate **0.05**, row and feature sampling **0.8**.

## LightGBM

| Parameter | Controls | Starting range / value |
| :--- | :--- | :--- |
| `n_estimators` | Boosting rounds | **300–2,000**, with early stopping |
| `learning_rate` | Each tree's contribution | **0.01–0.2** |
| `num_leaves` | Maximum leaves per tree | **7, 15, 31, 63** |
| `max_depth` | Tree depth | **4–10**, or **-1** (unlimited) |
| `min_child_samples` | Minimum observations per leaf | **10–100** |
| `min_child_weight` | Minimum Hessian per leaf | Usually leave at **0.001** initially |
| `subsample` | Fraction of sampled rows | **0.6–1.0** |
| `subsample_freq` | How often rows are resampled | **1** to resample each round; **0** disables it |
| `colsample_bytree` | Fraction of features per tree | **0.6–1.0** |
| `reg_alpha` | L1 penalty on leaf outputs | **0, 0.01, 0.1, 1, 10** |
| `reg_lambda` | L2 penalty on leaf outputs | **0, 0.1, 1, 10** |
| `min_split_gain` | Minimum gain required to split | **0** initially; scale-dependent |
| Early-stopping callback | Rounds without validation improvement | **50–100** |
| `max_bin` | Histogram resolution | **63, 127, 255** |

**Simple start:** **15 or 31** leaves, learning rate **0.05**, minimum leaf size **20**.

With a positive depth limit, keep `num_leaves <= 2**max_depth`.

## Random Forest

| Parameter | Controls | Starting range / value |
| :--- | :--- | :--- |
| `n_estimators` | Number of trees | **200–1,000** |
| `max_depth` | Tree depth | **5–30**, plus **None** (unlimited) |
| `min_samples_leaf` | Minimum observations per leaf | **1, 2, 5, 10, 20** |
| `min_samples_split` | Minimum observations needed to split | **2, 5, 10, 20** |
| `max_features` | Candidate features per split | **0.3–1.0** for regression; **"sqrt"** for classification |
| `bootstrap` | Sample rows with replacement | **True** |
| `max_samples` | Bootstrap sample size | **0.6–1.0** of training rows, or **None** (full size) |
| `max_leaf_nodes` | Maximum leaves per tree | Usually **None**; optionally **16–128** |
| `min_impurity_decrease` | Minimum improvement required to split | **0** initially; scale-dependent |

**Simple start:** **500** trees, `max_depth=None`, `min_samples_leaf=2`.

## Regression vs. classification

| Model | Regression | Classification |
| :--- | :--- | :--- |
| **XGBoost objective** | `reg:squarederror` | `binary:logistic` / `multi:softprob` |
| **LightGBM objective** | `regression` | `binary` / `multiclass` |
| **RF split criterion** | `squared_error` | `gini` / `entropy` / `log_loss` |
| **Common metrics** | MAE, RMSE, R² | Log loss, ROC-AUC, precision, recall, F1 |

For class imbalance: compare RF's `class_weight="balanced"`; for binary boosting, compare `scale_pos_weight=1` with `n_negative / n_positive`. Validate the choice.

## Quick tuning rules

| Situation | Try |
| :--- | :--- |
| **Overfitting** | Shallower trees, fewer leaves, larger minimum leaves, stronger penalties |
| **Underfitting** | Deeper trees, more leaves, weaker restrictions |
| **Lower learning rate** | More boosting rounds |
| **Small dataset** | XGBoost depth **3–5**; LightGBM **7–31** leaves |
| **Large dataset** | Consider minimum leaf sizes in the hundreds or thousands |

## Keep in mind

- **Leaf:** a final group in a tree. **Hessian:** loss curvature, not generally a sample count.
- **L1/L2:** penalize leaf outputs. RF uses tree-size restrictions instead.
- **Sampling:** RF normally samples rows with replacement; ordinary boosting subsampling does not.
- **Validation:** tune and early-stop on validation data; keep the test set separate. Use chronological splits for time series.
- **Utilities:** fix `random_state` (e.g., **42**); use `n_jobs` to control CPU workers.

---

**Documentation:** [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html) · [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html) · [RF regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) · [RF classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
