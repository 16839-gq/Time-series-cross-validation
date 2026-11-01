
from tscv import GapRollForward
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from utils import (
    validate_splits,
    cross_validate_regressor,
    plot_cv_splits
)

# Define time-series cross-validation
cv = GapRollForward(
    min_train_size=100,
    max_test_size=20,
    gap_size=5,
    roll_size=20
)

# Check folds
print(validate_splits(cv, X, y, min_gap=5))

# Evaluate model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=1.0)
)

results = cross_validate_regressor(model, X, y, cv, min_gap=5)
print(results[["fold", "MAE", "RMSE", "R2"]])

# Visualize splits
fig, ax = plot_cv_splits(cv, X)
