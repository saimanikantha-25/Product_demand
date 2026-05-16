# Product Demand Forecasting

A small project that **forecasts product demand** using Amazon product data. Because the dataset does not contain historical sales time series, **rating count** is used as a proxy for demand: products with more ratings are treated as higher demand.

## Project structure

```
product_demand/
├── data/
│   └── amazon.csv          # Amazon product data (required)
├── src/
│   ├── __init__.py
│   ├── config.py           # Paths and constants
│   ├── data_loader.py      # Load and clean CSV
│   ├── feature_engineering.py  # Demand level, price, category, text features
│   └── model.py            # Random Forest training and evaluation
├── app.py                  # Web GUI (Streamlit)
├── run_forecasting.py      # Run full pipeline (no Jupyter)
├── output/                 # Created when you run the script (plots)
├── product_demand_forecasting.ipynb  # Optional notebook
├── requirements.txt
└── README.md
```

## Setup

1. Ensure the Amazon data file is at `data/amazon.csv`.
2. Install dependencies:

   ```bash
   cd product_demand
   pip install -r requirements.txt
   ```

## Run with GUI (recommended)

From the `product_demand` folder:

```bash
streamlit run app.py
```

A browser window opens with a dashboard: overview metrics (products loaded, MAE, R², RMSE), numeric summary table, EDA charts (demand distribution, categories), and feature importance. Use the sidebar to change sample size and click **Run analysis** to refresh.

## Run without GUI (script only)

```bash
python run_forecasting.py
```

Runs the full pipeline and saves plots to the `output/` folder (`eda_demand.png`, `eda_categories.png`, `feature_importance.png`).

## Run with Jupyter (optional)

Start Jupyter and open `product_demand_forecasting.ipynb`:

```bash
jupyter notebook product_demand_forecasting.ipynb
```

Run the notebook from the `product_demand` directory so that `src` can be imported.

## What the notebook does

1. **Load and clean** – Reads `amazon.csv`, parses prices (₹), discount (%), and rating count.
2. **EDA** – Distribution of demand (rating count), demand levels (low/medium/high), price vs demand, top categories.
3. **Feature engineering** – Log price, discount flag, discount %, rating, category (encoded), log description length.
4. **Model** – Trains a Random Forest regressor to predict `rating_count`; reports MAE, RMSE, R² and feature importance.

## Data

- **Target:** `rating_count` (proxy for demand).
- **Features:** discounted price (log), discount %, rating, category, description length (log).

For real time-series demand forecasting you would need historical sales data; this project demonstrates the workflow using the available Amazon dataset.
