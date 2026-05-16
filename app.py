"""
Product Demand Forecasting — Professional GUI (Streamlit).

Run from product_demand folder:
    streamlit run app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import streamlit as st
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

# Page config
st.set_page_config(
    page_title="Product Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background: linear-gradient(160deg, #f0f4f8 0%, #e2e8f0 35%, #cbd5e1 70%, #94a3b8 100%);
        background-attachment: fixed;
    }
    /* Optional: subtle pattern overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
                          radial-gradient(circle at 80% 20%, rgba(30, 58, 95, 0.05) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    /* Main content area - slight card effect for readability */
    section.main .block-container {
        background: rgba(255, 255, 255, 0.85);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        margin: 1rem 0;
        position: relative;
        z-index: 1;
    }
    /* Sidebar: all text white */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #1e40af 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: rgba(255,255,255,0.95) !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: rgba(255,255,255,0.85) !important;
    }
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] div[data-testid="stSlider"] + div {
        color: rgba(255,255,255,0.95) !important;
    }
    /* Remove hover/focus animation — sidebar buttons look the same with or without mouse */
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] [data-baseweb="button"] {
        transition: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] .stButton > button:focus,
    section[data-testid="stSidebar"] .stButton > button:focus-visible,
    section[data-testid="stSidebar"] .stButton > button:active {
        color: rgba(255,255,255,0.95) !important;
        background-color: rgba(255,255,255,0.18) !important;
        border-color: rgba(255,255,255,0.4) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"],
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:hover,
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:focus,
    section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:active {
        background-color: #2563eb !important;
        color: #fff !important;
        border-color: #2563eb !important;
    }
    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 50%, #3b82f6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 14px rgba(30, 58, 95, 0.25);
    }
    .hero h1 {
        color: white !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }
    .hero p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 0.95rem !important;
        margin: 0.4rem 0 0 0 !important;
    }
    /* Main content text — visible on background */
    section.main .block-container,
    section.main .block-container p,
    section.main .block-container span,
    section.main .block-container li,
    section.main [data-testid="stMarkdown"] {
        color: #334155 !important;
    }
    section.main .stSubheader,
    section.main h2, section.main h3 {
        color: #1e3a5f !important;
    }
    section.main [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }
    section.main [data-testid="stMetricValue"] {
        color: #1e3a5f !important;
    }
    /* Section titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e3a5f;
        margin: 1.25rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #cbd5e1;
    }
    /* Metric cards */
    .metric-box {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-box .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
    }
    .metric-box .label {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }
    /* Insights callout */
    .insights-callout {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border-left: 4px solid #059669;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .insights-callout strong { color: #065f46; }
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #cbd5e1;
        font-size: 0.8rem;
        color: #475569;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def run_analysis_amazon(max_rows: int):
    """Load Amazon data, build features, train model; return dfs, metrics, figures."""
    df_raw = load_data()
    df = get_analysis_sample(df_raw, max_rows=max_rows)

    df_eda = add_demand_level(df)
    df_eda = add_category_features(df_eda)

    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
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

    fig2, ax = plt.subplots(figsize=(8, 5))
    top_cats = df_eda["category_top"].value_counts().head(10)
    top_cats.plot(kind="barh", ax=ax)
    ax.set_title("Top 10 categories by product count")
    ax.set_xlabel("Count")
    plt.tight_layout()

    df = add_price_features(df)
    df = add_category_features(df)
    df = add_text_length_features(df)
    X, y, feature_names = build_feature_matrix(df)
    model, X_test, y_test, y_pred, metrics = train_demand_model(X, y)
    importance = get_feature_importance(model, feature_names)

    fig3, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance for demand forecast")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    summary_df = df[["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]].describe()
    top_feature = importance.index[-1] if len(importance) else "—"

    return {
        "df_raw": df_raw,
        "df": df,
        "df_eda": df_eda,
        "summary_df": summary_df,
        "metrics": metrics,
        "feature_names": feature_names,
        "importance": importance,
        "top_feature": top_feature,
        "fig_demand": fig1,
        "fig_categories": fig2,
        "fig_importance": fig3,
    }


@st.cache_data(ttl=600)
def run_analysis_store_sales(max_rows: int):
    """Load store sales data, build features, train model; return dfs, metrics, figures."""
    df_raw = load_store_sales_data()
    df = get_store_sales_analysis_sample(df_raw, max_rows=max_rows)

    df = add_time_features_store_sales(df)
    df_eda = add_store_sales_demand_level(df)
    df = add_store_item_encoded(df)

    fig1, axes = plt.subplots(2, 2, figsize=(10, 8))
    df_eda["sales"].hist(bins=50, ax=axes[0, 0], edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Distribution of Sales")
    axes[0, 0].set_xlabel("Sales")

    df_eda["demand_level"].value_counts().sort_index().plot(kind="bar", ax=axes[0, 1], color="steelblue")
    axes[0, 1].set_title("Demand level counts (store sales)")
    axes[0, 1].set_xlabel("Demand level")

    df_eda.boxplot(column="sales", by="demand_level", ax=axes[1, 0])
    axes[1, 0].set_title("Sales by demand level")
    plt.suptitle("")

    sales_by_store = df_eda.groupby("store")["sales"].sum()
    sales_by_store.head(10).plot(kind="barh", ax=axes[1, 1])
    axes[1, 1].set_title("Total sales by store (top 10)")
    axes[1, 1].set_xlabel("Total sales")
    plt.tight_layout()

    X, y, feature_names = build_feature_matrix_store_sales(df)
    model, X_test, y_test, y_pred, metrics = train_demand_model(X, y)
    importance = get_feature_importance(model, feature_names)

    fig2, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax)
    ax.set_title("Feature importance (store sales demand)")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    summary_df = df[["store", "item", "sales", "day_of_week", "month", "year"]].describe()
    top_feature = importance.index[-1] if len(importance) else "—"

    return {
        "df_raw": df_raw,
        "df": df,
        "df_eda": df_eda,
        "summary_df": summary_df,
        "metrics": metrics,
        "feature_names": feature_names,
        "importance": importance,
        "top_feature": top_feature,
        "fig_demand": fig1,
        "fig_importance": fig2,
    }


def main():
    # Hero header
    st.markdown("""
    <div class="hero">
        <h1>📊 Product Demand Forecasting</h1>
        <p>Choose the dataset to train (Amazon or Store sales), then run analysis and explore metrics, EDA, and the model in the tabs below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: click to select dataset; slider and options only after selection
    if "dataset_to_train" not in st.session_state:
        st.session_state["dataset_to_train"] = None
    if "use_all_records" not in st.session_state:
        st.session_state["use_all_records"] = False

    with st.sidebar:
        st.markdown("<span style='color: rgba(255,255,255,0.95); font-weight: 600;'>Choose dataset</span>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Amazon" + (" ✓" if st.session_state["dataset_to_train"] == "Amazon" else ""), type="primary" if st.session_state["dataset_to_train"] == "Amazon" else "secondary", use_container_width=True):
                st.session_state["dataset_to_train"] = "Amazon"
                st.rerun()
        with col_b:
            if st.button("Store sales" + (" ✓" if st.session_state["dataset_to_train"] == "Store sales" else ""), type="primary" if st.session_state["dataset_to_train"] == "Store sales" else "secondary", use_container_width=True):
                st.session_state["dataset_to_train"] = "Store sales"
                st.rerun()

        dataset_to_train = st.session_state["dataset_to_train"]

        # Show slider and "Use all records" only when a dataset is selected
        if dataset_to_train is not None:
            if st.button("✓ Use all records" if st.session_state["use_all_records"] else "Use all records (no sampling)", use_container_width=True):
                st.session_state["use_all_records"] = not st.session_state["use_all_records"]
                st.rerun()
            use_all_records = st.session_state["use_all_records"]
            if dataset_to_train == "Amazon":
                max_rows = st.slider("Max Amazon products", min_value=500, max_value=5000, value=3000, step=500)
            else:
                max_rows = st.slider("Max store sales rows", min_value=10000, max_value=80000, value=50000, step=10000)
            run_btn = st.button("🔄 Run analysis", type="primary", use_container_width=True)
        else:
            use_all_records = False
            max_rows = 3000
            run_btn = False

        st.divider()
        st.caption("**Amazon**: rating_count = demand. Features: price, discount, rating, category, text length.")
        st.caption("**Store sales**: date, store, item, sales. Features: store, item, day_of_week, month, year.")

    effective_max = 10**6 if use_all_records else max_rows

    if dataset_to_train is None:
        st.info("Click **Amazon** or **Store sales** in the sidebar to choose a dataset, then set sample size and run analysis.")
        st.markdown('<div class="footer">Product Demand Forecasting · Choose a dataset to start</div>', unsafe_allow_html=True)
        return

    # Run only the selected dataset when Run is clicked
    if run_btn:
        with st.spinner(f"Loading and training {dataset_to_train}..."):
            try:
                if dataset_to_train == "Amazon":
                    st.session_state["results_amazon"] = run_analysis_amazon(effective_max)
                else:
                    st.session_state["results_store_sales"] = run_analysis_store_sales(effective_max)
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()

    # Get results for the selected dataset only
    if dataset_to_train == "Amazon":
        r = st.session_state.get("results_amazon")
    else:
        r = st.session_state.get("results_store_sales")

    if r is None:
        st.info(f"Select **{dataset_to_train}** above and click **Run analysis** to train the model and see results.")
        st.markdown('<div class="footer">Product Demand Forecasting · Choose a dataset and run analysis</div>', unsafe_allow_html=True)
        return

    # Tabs for the selected dataset only
    tab_overview, tab_data, tab_eda, tab_model = st.tabs(["📈 Overview", "📋 Data", "🔍 EDA", "🤖 Model"])

    with tab_overview:
        st.markdown(f'<p class="section-title">Key metrics — {dataset_to_train}</p>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Rows loaded", f"{r['df_raw'].shape[0]:,}")
        with c2:
            st.metric("Sample size", f"{r['df'].shape[0]:,}")
        with c3:
            mae = r["metrics"]["MAE"]
            st.metric("MAE", f"{mae:,.0f}" if mae >= 1 else f"{mae:.2f}")
        with c4:
            rmse = r["metrics"]["RMSE"]
            st.metric("RMSE", f"{rmse:,.0f}" if rmse >= 1 else f"{rmse:.2f}")
        with c5:
            st.metric("R²", f"{r['metrics']['R2']:.2%}")

        with st.expander("What do MAE, RMSE, and R² mean?"):
            st.markdown("""
            - **MAE** (Mean Absolute Error): Average prediction error. Lower is better.
            - **RMSE** (Root Mean Squared Error): Penalizes large errors more. Lower is better.
            - **R²** (R-squared): Share of variance explained by the model (0–100%). Higher is better.
            """)

        st.markdown('<p class="section-title">Key insights</p>', unsafe_allow_html=True)
        r2_pct = r["metrics"]["R2"] * 100
        mae_fmt = f"{r['metrics']['MAE']:,.0f}" if r["metrics"]["MAE"] >= 1 else f"{r['metrics']['MAE']:.2f}"
        st.markdown(f"""
        <div class="insights-callout">
            <strong>{dataset_to_train}</strong><br>
            • The model explains <strong>{r2_pct:.1f}%</strong> of the variation in demand (R²).<br>
            • MAE = <strong>{mae_fmt}</strong>. Top feature: <strong>{r['top_feature']}</strong>.
        </div>
        """, unsafe_allow_html=True)

    with tab_data:
        st.markdown(f'<p class="section-title">{dataset_to_train} — numeric summary</p>', unsafe_allow_html=True)
        st.dataframe(r["summary_df"].round(2), use_container_width=True)
        st.markdown('<p class="section-title">Sample rows</p>', unsafe_allow_html=True)
        if dataset_to_train == "Amazon":
            display_cols = ["product_name", "category", "discounted_price", "rating", "rating_count"]
            available = [c for c in display_cols if c in r["df"].columns]
            st.dataframe(r["df"][available].head(20), use_container_width=True)
            csv = r["df"][["discounted_price", "actual_price", "discount_percentage", "rating", "rating_count"]].to_csv(index=False)
            st.download_button("Download sample CSV", data=csv, file_name="amazon_demand_sample.csv", mime="text/csv")
        else:
            st.dataframe(r["df"][["date", "store", "item", "sales", "day_of_week", "month", "year"]].head(20), use_container_width=True)
            csv = r["df"][["date", "store", "item", "sales"]].to_csv(index=False)
            st.download_button("Download sample CSV", data=csv, file_name="store_sales_sample.csv", mime="text/csv")

    with tab_eda:
        st.markdown(f'<p class="section-title">{dataset_to_train} — demand & EDA</p>', unsafe_allow_html=True)
        st.pyplot(r["fig_demand"])
        if dataset_to_train == "Amazon":
            st.pyplot(r["fig_categories"])

    with tab_model:
        st.markdown('<p class="section-title">Feature importance</p>', unsafe_allow_html=True)
        st.pyplot(r["fig_importance"])
        st.caption("Random Forest: which features drive demand most.")

    st.markdown(f'<div class="footer">Product Demand Forecasting · {dataset_to_train} · Model: Random Forest</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
