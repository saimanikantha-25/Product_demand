"""Feature engineering for product demand forecasting."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import TARGET_COLUMN, DEMAND_LEVEL_LABELS, STORE_SALES_TARGET


def add_demand_level(df, target_col=TARGET_COLUMN, n_quantiles=3):
    """
    Add a categorical demand level (low / medium / high) based on target quantiles.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    n_quantiles : int
        Number of bins (e.g. 3 -> low, medium, high).

    Returns
    -------
    pd.DataFrame
        Copy of df with new column 'demand_level'.
    """
    df = df.copy()
    q = np.linspace(0, 1, n_quantiles + 1)[1:-1]
    thresholds = df[target_col].quantile(q).values
    labels = DEMAND_LEVEL_LABELS[:n_quantiles] if n_quantiles <= len(DEMAND_LEVEL_LABELS) else [str(i) for i in range(n_quantiles)]

    def _level(x):
        for i, t in enumerate(thresholds):
            if x <= t:
                return labels[i]
        return labels[-1]

    df["demand_level"] = df[target_col].apply(_level)
    return df


def add_price_features(df):
    """
    Add price-related features: log price, discount flag, price tier.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["log_discounted_price"] = np.log1p(df["discounted_price"].clip(lower=1))
    df["has_discount"] = (df["discount_percentage"].fillna(0) > 0).astype(int)
    df["discount_pct"] = df["discount_percentage"].fillna(0)
    return df


def add_category_features(df, top_n=15):
    """
    Add top-level category and one-hot / label-encoded category feature.

    Expects 'category' column with format like 'A|B|C|D'.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    # Top-level category (first part before |)
    df["category_top"] = df["category"].fillna("").apply(lambda x: str(x).split("|")[0] if "|" in str(x) else str(x))
    df["category_top"] = df["category_top"].replace("", "Unknown")

    # Limit to top_n categories for encoding
    top_cats = df["category_top"].value_counts().head(top_n).index.tolist()
    df["category_top_limited"] = df["category_top"].apply(lambda x: x if x in top_cats else "Other")

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category_top_limited"].astype(str))
    return df


def add_text_length_features(df, text_col="about_product"):
    """
    Add length of product description as a simple text feature.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    col = text_col if text_col in df.columns else "product_name"
    df["text_length"] = df[col].fillna("").astype(str).str.len()
    df["log_text_length"] = np.log1p(df["text_length"])
    return df


def build_feature_matrix(df):
    """
    Build numeric feature matrix X and target y for modeling.

    Features: log_discounted_price, has_discount, discount_pct, rating,
              category_encoded, log_text_length.

    Expects df to already have price, category, and text-length features
    (run add_price_features, add_category_features, add_text_length_features first).

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    feature_names : list of str
    """
    required = [
        "log_discounted_price",
        "has_discount",
        "discount_pct",
        "rating",
        "category_encoded",
        "log_text_length",
        TARGET_COLUMN,
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Run add_price_features, add_category_features, add_text_length_features first.")

    feature_cols = [c for c in required if c != TARGET_COLUMN]
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values
    return X, y, feature_cols


# ----- Store sales (date, store, item, sales) -----


def add_time_features_store_sales(df, date_col="date"):
    """
    Add time-based features from date: day_of_week, month, year.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["month"] = df[date_col].dt.month
    df["year"] = df[date_col].dt.year
    return df


def add_store_sales_demand_level(df, target_col=STORE_SALES_TARGET, n_quantiles=3):
    """
    Add categorical demand level (low / medium / high) for store sales.

    Returns
    -------
    pd.DataFrame
    """
    return add_demand_level(df, target_col=target_col, n_quantiles=n_quantiles)


def add_store_item_encoded(df):
    """
    Add label-encoded store and item for modeling.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    le_store = LabelEncoder()
    le_item = LabelEncoder()
    df["store_encoded"] = le_store.fit_transform(df["store"].astype(str))
    df["item_encoded"] = le_item.fit_transform(df["item"].astype(str))
    return df


def build_feature_matrix_store_sales(df):
    """
    Build feature matrix X and target y for store sales model.

    Expects df with: date, store, item, sales, and time/encoded features from
    add_time_features_store_sales and add_store_item_encoded.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    feature_names : list of str
    """
    required = [
        "store_encoded",
        "item_encoded",
        "day_of_week",
        "month",
        "year",
        STORE_SALES_TARGET,
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(
                f"Missing column '{c}'. Run add_time_features_store_sales and add_store_item_encoded first."
            )
    feature_cols = [c for c in required if c != STORE_SALES_TARGET]
    X = df[feature_cols].values
    y = df[STORE_SALES_TARGET].values
    return X, y, feature_cols
