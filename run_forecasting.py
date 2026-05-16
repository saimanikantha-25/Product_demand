"""
Product Demand Forecasting — run without Jupyter.

Usage (from product_demand folder):
    python run_forecasting.py

Saves plots to output/ and prints metrics to the console.
"""

import sys
from pathlib import Path

# Ensure we can import src when run from product_demand
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import (
    load_data,
    get_analysis_sample,
    load_store_sales_data,
    get_store_sales_analysis_sample,
)
from src.feature_engineering import (
    add_demand_level,
    add_price_features,
    add_category_features,
    add_text_length_features,
    build_feature_matrix,
    add_time_features_store_sales,
    add_store_sales_demand_level,
    add_store_item_encoded,
    build_feature_matrix_store_sales,
)
from src.model import train_demand_model, get_feature_importance


def main():
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Product Demand Forecasting")
    print("=" * 60)

    # 1. Load and clean data
    print("\n1. Loading data...")
    df_raw = load_data()
    print(f"   Raw shape: {df_raw.shape}")

    df = get_analysis_sample(df_raw, max_rows=3000)
    print(f"   Analysis sample shape: {df.shape}")

    print("\n   Numeric summary:")
    print(df[["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]].describe().to_string())

    # 2. EDA — demand level and plots
    print("\n2. Exploratory analysis...")
    df_eda = add_demand_level(df)
    df_eda = add_category_features(df_eda)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    df_eda["rating_count"].hist(bins=50, ax=axes[0, 0], edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Demand (rating_count)")
    axes[0, 0].set_xlabel("Rating count")

    df_eda["demand_level"].value_counts().sort_index().plot(kind="bar", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Demand level counts")
    axes[0, 1].set_xlabel("Demand level")

    df_eda.plot.scatter(x="discounted_price", y="rating_count", alpha=0.4, ax=axes[1, 0])
    axes[1, 0].set_title("Price vs Demand")

    df_eda.boxplot(column="rating_count", by="demand_level", ax=axes[1, 1])
    axes[1, 1].set_title("Demand by level")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_demand.png", dpi=120)
    plt.close()
    print(f"   Saved {output_dir / 'eda_demand.png'}")

    top_cats = df_eda["category_top"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    top_cats.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 categories by product count")
    ax.set_xlabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_categories.png", dpi=120)
    plt.close()
    print(f"   Saved {output_dir / 'eda_categories.png'}")

    # 3. Feature engineering and model
    print("\n3. Feature engineering and model training...")
    df = add_price_features(df)
    df = add_category_features(df)
    df = add_text_length_features(df)
    X, y, feature_names = build_feature_matrix(df)
    print(f"   Features: {feature_names}")
    print(f"   Samples: {len(X)}")

    model, X_test, y_test, y_pred, metrics = train_demand_model(X, y)
    print("\n4. Demand forecasting metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    importance = get_feature_importance(model, feature_names)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance for demand forecast")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=120)
    plt.close()
    print(f"\n   Saved {output_dir / 'feature_importance.png'}")

    # ----- Store sales pipeline -----
    print("\n" + "=" * 60)
    print("Store Sales Demand Forecasting")
    print("=" * 60)

    print("\n1. Loading store sales data...")
    df_sales_raw = load_store_sales_data()
    print(f"   Raw shape: {df_sales_raw.shape}")

    df_sales = get_store_sales_analysis_sample(df_sales_raw, max_rows=50000)
    print(f"   Analysis sample shape: {df_sales.shape}")

    print("\n   Numeric summary (sales):")
    print(df_sales[["store", "item", "sales"]].describe().to_string())

    print("\n2. Exploratory analysis (store sales)...")
    df_sales = add_time_features_store_sales(df_sales)
    df_sales_eda = add_store_sales_demand_level(df_sales)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    df_sales_eda["sales"].hist(bins=50, ax=axes[0, 0], edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Sales")
    axes[0, 0].set_xlabel("Sales")

    df_sales_eda["demand_level"].value_counts().sort_index().plot(
        kind="bar", ax=axes[0, 1], color="steelblue"
    )
    axes[0, 1].set_title("Demand level counts (store sales)")
    axes[0, 1].set_xlabel("Demand level")

    df_sales_eda.boxplot(column="sales", by="demand_level", ax=axes[1, 0])
    axes[1, 0].set_title("Sales by demand level")
    plt.suptitle("")

    sales_by_store = df_sales_eda.groupby("store")["sales"].sum()
    sales_by_store.head(10).plot(kind="barh", ax=axes[1, 1])
    axes[1, 1].set_title("Total sales by store (top 10)")
    axes[1, 1].set_xlabel("Total sales")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_store_sales_demand.png", dpi=120)
    plt.close()
    print(f"   Saved {output_dir / 'eda_store_sales_demand.png'}")

    print("\n3. Feature engineering and model (store sales)...")
    df_sales = add_store_item_encoded(df_sales)
    X_sales, y_sales, feature_names_sales = build_feature_matrix_store_sales(df_sales)
    print(f"   Features: {feature_names_sales}")
    print(f"   Samples: {len(X_sales)}")

    model_sales, X_test_s, y_test_s, y_pred_s, metrics_sales = train_demand_model(
        X_sales, y_sales
    )
    print("\n4. Store sales forecasting metrics:")
    for k, v in metrics_sales.items():
        print(f"   {k}: {v:.4f}")

    importance_sales = get_feature_importance(model_sales, feature_names_sales)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance_sales.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance (store sales demand)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_store_sales.png", dpi=120)
    plt.close()
    print(f"\n   Saved {output_dir / 'feature_importance_store_sales.png'}")

    print("\n" + "=" * 60)
    print("Done. Check the output/ folder for all plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
