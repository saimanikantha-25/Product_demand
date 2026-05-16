"""Configuration for product demand forecasting."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "amazon.csv"
STORE_SALES_DATA_PATH = DATA_DIR / "store_sales.csv"

# Target and demand (Amazon)
TARGET_COLUMN = "rating_count"
DEMAND_LEVEL_LABELS = ["low", "medium", "high"]

# Store sales
STORE_SALES_TARGET = "sales"

# Model
RANDOM_STATE = 42
TEST_SIZE = 0.2
