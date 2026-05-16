"""Train and evaluate demand forecasting model."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import TARGET_COLUMN, RANDOM_STATE, TEST_SIZE


def train_demand_model(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Train a Random Forest regressor to predict demand (rating_count).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    test_size : float
    random_state : int

    Returns
    -------
    model : RandomForestRegressor
    X_test, y_test : arrays
    y_pred : np.ndarray
        Predictions on test set.
    metrics : dict
        MAE, RMSE, R2 on test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=12)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }
    return model, X_test, y_test, y_pred, metrics


def get_feature_importance(model, feature_names):
    """
    Return a pandas Series of feature importances from the trained model.

    Parameters
    ----------
    model : RandomForestRegressor
    feature_names : list of str

    Returns
    -------
    pd.Series
        Index = feature names, values = importance.
    """
    imp = model.feature_importances_
    return pd.Series(imp, index=feature_names).sort_values(ascending=True)
