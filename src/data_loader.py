"""Load and clean Amazon product data for demand forecasting."""

import re
import pandas as pd

from .config import RAW_DATA_PATH, STORE_SALES_DATA_PATH, TARGET_COLUMN


def _parse_currency(value):
    """Parse Indian rupee string (e.g. '₹1,099' or '₹399') to float."""
    if pd.isna(value):
        return float("nan")
    s = str(value).strip().replace("₹", "").replace(",", "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_percent(value):
    """Parse percentage string (e.g. '64%') to float."""
    if pd.isna(value):
        return float("nan")
    s = str(value).strip().replace("%", "").strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _parse_count(value):
    """Parse count string (e.g. '24,269' or '1,79,691') to int."""
    if pd.isna(value):
        return 0
    s = str(value).strip().replace(",", "").strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def load_data(path=None):
    """
    Load Amazon product CSV and clean numeric columns.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with parsed discounted_price, actual_price,
        discount_percentage, rating, rating_count.
    """
    path = path or RAW_DATA_PATH
    df = pd.read_csv(path)

    # Parse price and numeric columns
    df["discounted_price"] = df["discounted_price"].apply(_parse_currency)
    df["actual_price"] = df["actual_price"].apply(_parse_currency)
    df["discount_percentage"] = df["discount_percentage"].apply(_parse_percent)
    df["rating_count"] = df["rating_count"].apply(_parse_count)

    # Ensure rating is numeric
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Drop rows where target or key features are missing/invalid
    df = df.dropna(subset=["discounted_price", "rating", "rating_count"])
    df = df[df["rating_count"] >= 0]
    df = df[df["discounted_price"] > 0]

    return df


def get_analysis_sample(df, max_rows=3000, random_state=42):
    """
    Return a random sample of the dataframe for faster analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset from load_data().
    max_rows : int
        Maximum number of rows to keep. If df has fewer rows, returns df.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe.
    """
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


def load_store_sales_data(path=None):
    """
    Load store sales CSV (date, store, item, sales) and parse dates.

    Returns
    -------
    pd.DataFrame
        Dataframe with date (datetime), store, item, sales.
    """
    path = path or STORE_SALES_DATA_PATH
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "store", "item", "sales"])
    df = df[df["sales"] >= 0]
    return df


def get_store_sales_analysis_sample(df, max_rows=50000, random_state=42):
    """
    Return a random sample of store sales for faster analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset from load_store_sales_data().
    max_rows : int
        Maximum number of rows to keep.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Sampled dataframe.
    """
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()
