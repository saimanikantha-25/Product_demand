"""Product demand forecasting package."""

from .config import TARGET_COLUMN, DEMAND_LEVEL_LABELS, RAW_DATA_PATH
from .data_loader import load_data, get_analysis_sample
from .feature_engineering import (
    add_demand_level,
    add_price_features,
    add_category_features,
    add_text_length_features,
    build_feature_matrix,
)
from .model import train_demand_model, get_feature_importance

__all__ = [
    "TARGET_COLUMN",
    "DEMAND_LEVEL_LABELS",
    "RAW_DATA_PATH",
    "load_data",
    "get_analysis_sample",
    "add_demand_level",
    "add_price_features",
    "add_category_features",
    "add_text_length_features",
    "build_feature_matrix",
    "train_demand_model",
    "get_feature_importance",
]
