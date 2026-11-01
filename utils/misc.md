# Tree Models · Quick Reference

**Random Forest · XGBoost · LightGBM**

All three support **regression and classification**. Ranges below are suggested starting points, not fixed limits or library defaults.

## At a glance

| Model | How it works |
| :--- | :--- |
| **Random Forest** | Trains separate trees and averages their predictions. |
| **XGBoost** | Adds trees sequentially to improve predictions. |
| **LightGBM** | Also uses boosting; grows the most useful leaf first and uses histogram bins for speed. |

**↑ / ↓** show typical effects with other settings fixed. Special values such as `0`, `-1`, and `None` follow their stated meanings.

## XGBoost

| Parameter | Controls | Starting range / value | Higher ↑ / lower ↓ |
| :--- | :--- | :--- | :--- |
| `n_estimators` | Boosting rounds | **300–2,000**, with early stopping | ↑ More rounds allowed, more capacity/time; ↓ less |
| `learning_rate` | Each tree's contribution | **0.01–0.2** | ↑ Larger updates, faster fitting; ↓ smaller updates, more rounds needed |
| `max_depth` | Tree depth | **3–8** | ↑ More complex trees, more overfitting risk; ↓ simpler trees |
| `min_child_weight` | Minimum Hessian per leaf | **1–10** | ↑ Fewer eligible splits; ↓ more flexible trees |
| `subsample` | Fraction of rows per round | **0.6–1.0** | ↑ More rows, less sampling randomness; ↓ fewer rows, more randomness |
| `colsample_bytree` | Fraction of features per tree | **0.6–1.0** | ↑ More features available; ↓ more randomness, less computation |
| `reg_alpha` | L1 penalty on leaf outputs | **0, 0.01, 0.1, 1, 10** | ↑ More shrinkage and zero leaf outputs; ↓ weaker penalty |
| `reg_lambda` | L2 penalty on leaf outputs | **0.1, 1, 10** | ↑ Smaller leaf updates; ↓ larger updates |
| `gamma` | Minimum gain required to split | **0** initially; scale-dependent | ↑ Fewer splits; ↓ easier splitting |
| `early_stopping_rounds` | Rounds without validation improvement | **50–100** | ↑ Wait longer for improvement; ↓ stop sooner |
| `max_bin` | Histogram resolution | **128 or 256**, with `tree_method="hist"` | ↑ Finer splits, more computation; ↓ coarser splits, faster |
| `max_leaves` | Maximum leaves | Usually **0** (unlimited); optional with histogram trees | ↑ More detailed trees; ↓ simpler trees |

**Simple start:** depth **4**, learning rate **0.05**, row and feature sampling **0.8**.

## LightGBM

| Parameter | Controls | Starting range / value | Higher ↑ / lower ↓ |
| :--- | :--- | :--- | :--- |
| `n_estimators` | Boosting rounds | **300–2,000**, with early stopping | ↑ More rounds allowed, more capacity/time; ↓ less |
| `learning_rate` | Each tree's contribution | **0.01–0.2** | ↑ Larger updates, faster fitting; ↓ smaller updates, more rounds needed |
| `num_leaves` | Maximum leaves per tree | **7, 15, 31, 63** | ↑ More complex trees, more overfitting risk; ↓ simpler trees |
| `max_depth` | Tree depth | **4–10**, or **-1** (unlimited) | ↑ More complex trees, more overfitting risk; ↓ simpler trees |
| `min_child_samples` | Minimum observations per leaf | **10–100** | ↑ Larger leaves, smoother fit; ↓ smaller leaves, more detail |
| `min_child_weight` | Minimum Hessian per leaf | Usually leave at **0.001** initially | ↑ Fewer eligible splits; ↓ more flexible trees |
| `subsample` | Fraction of sampled rows | **0.6–1.0** | ↑ More rows, less sampling randomness; ↓ fewer rows, more randomness |
| `subsample_freq` | Rounds between row resamples | **1** to resample each round; **0** disables it | ↑ Resample less often; ↓ more often |
| `colsample_bytree` | Fraction of features per tree | **0.6–1.0** | ↑ More features available; ↓ more randomness, less computation |
| `reg_alpha` | L1 penalty on leaf outputs | **0, 0.01, 0.1, 1, 10** | ↑ More shrinkage and zero leaf outputs; ↓ weaker penalty |
| `reg_lambda` | L2 penalty on leaf outputs | **0, 0.1, 1, 10** | ↑ Smaller leaf updates; ↓ larger updates |
| `min_split_gain` | Minimum gain required to split | **0** initially; scale-dependent | ↑ Fewer splits; ↓ easier splitting |
| Early-stopping callback | Rounds without validation improvement | **50–100** | ↑ Wait longer for improvement; ↓ stop sooner |
| `max_bin` | Histogram resolution | **63, 127, 255** | ↑ Finer splits, more computation; ↓ coarser splits, faster |

**Simple start:** **15 or 31** leaves, learning rate **0.05**, minimum leaf size **20**.

With a positive depth limit, keep `num_leaves <= 2**max_depth`.

## Random Forest

| Parameter | Controls | Starting range / value | Higher ↑ / lower ↓ |
| :--- | :--- | :--- | :--- |
| `n_estimators` | Number of trees | **200–1,000** | ↑ More stable averaging, slower; ↓ faster, less stable |
| `max_depth` | Tree depth | **5–30**, plus **None** (unlimited) | ↑ More complex trees, more overfitting risk; ↓ simpler trees |
| `min_samples_leaf` | Minimum observations per leaf | **1, 2, 5, 10, 20** | ↑ Larger leaves, smoother predictions; ↓ smaller leaves, more detail |
| `min_samples_split` | Minimum observations needed to split | **2, 5, 10, 20** | ↑ Fewer nodes can split; ↓ more detailed trees |
| `max_features` | Candidate features per split | **0.3–1.0** for regression; **"sqrt"** for classification | ↑ More choices, usually more similar trees; ↓ more tree diversity |
| `bootstrap` | Sample rows with replacement | **True** | True: sample with replacement; False: use all rows |
| `max_samples` | Bootstrap sample size | **0.6–1.0** of training rows, or **None** (full size) | ↑ More row draws per tree; ↓ fewer draws, faster training |
| `max_leaf_nodes` | Maximum leaves per tree | Usually **None**; optionally **16–128** | ↑ More detailed trees; ↓ simpler trees |
| `min_impurity_decrease` | Minimum improvement required to split | **0** initially; scale-dependent | ↑ Fewer splits; ↓ easier splitting |

**Simple start:** **500** trees, `max_depth=None`, `min_samples_leaf=2`.

## Regression vs. classification

| Model | Regression | Classification |
| :--- | :--- | :--- |
| **XGBoost objective** | `reg:squarederror` | `binary:logistic` / `multi:softprob` |
| **LightGBM objective** | `regression` | `binary` / `multiclass` |
| **RF split criterion** | `squared_error` | `gini` / `entropy` / `log_loss` |
| **Common metrics** | MAE, RMSE, R² | Log loss, ROC-AUC, precision, recall, F1 |

For class imbalance: compare RF's `class_weight="balanced"`; for binary boosting, compare `scale_pos_weight=1` with `n_negative / n_positive`. Higher `scale_pos_weight` puts more emphasis on positive examples; lower values put less. Validate the choice.

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
- **Utilities:** a larger `random_state` is not better; fix one seed. More `n_jobs` workers allow more parallel work but use more resources.

---

**Documentation:** [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html) · [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html) · [RF regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) · [RF classification](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
