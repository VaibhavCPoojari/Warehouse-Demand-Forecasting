import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta
from io import StringIO

st.set_page_config(page_title="Inventory Demand Optimizer", layout="wide")

st.title("📦 Retail Inventory Demand Optimization")

model_path = "xgb_demand_model.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure it exists.")
    st.stop()
model = joblib.load(model_path)

csv_path = "retail_inventory_dataset.csv"
if not os.path.exists(csv_path):
    st.error(f"Dataset file not found at {csv_path}. Please ensure it exists.")
    st.stop()
df = pd.read_csv(csv_path)

# Parse date and extract features
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.weekday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Feature Engineering
if 'lag_1' not in df.columns:
    df['lag_1'] = df.sort_values('date').groupby('product_id')['units_sold'].transform(lambda x: x.shift(1))
if 'lag_7' not in df.columns:
    df['lag_7'] = df.sort_values('date').groupby('product_id')['units_sold'].transform(lambda x: x.shift(7))
if 'rolling_mean_7' not in df.columns:
    df['rolling_mean_7'] = df.sort_values('date').groupby('product_id')['units_sold'].transform(lambda x: x.shift(1).rolling(window=7).mean())

required_features = [
    'price', 'stock_on_hand', 'promotion_flag', 'day_of_week',
    'lag_1', 'lag_7', 'rolling_mean_7', 'day', 'month', 'is_weekend'
]

missing_cols = [col for col in required_features if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset even after processing: {missing_cols}")
    st.stop()

df = df.dropna(subset=required_features)

# Sidebar Filters
with st.sidebar:
    st.header("🔍 Filter Options")
    store_filter = st.selectbox("Select Store", options=['All'] + sorted(df['store_id'].unique().tolist()))
    product_filter = st.selectbox("Select Product", options=['All'] + sorted(df['product_id'].unique().tolist()))
    date_range = st.date_input("Select Date Range", [df['date'].min(), df['date'].max()])

# Apply Filters
filtered_df = df.copy()
if store_filter != 'All':
    filtered_df = filtered_df[filtered_df['store_id'] == store_filter]
if product_filter != 'All':
    filtered_df = filtered_df[filtered_df['product_id'] == product_filter]
filtered_df = filtered_df[(filtered_df['date'] >= pd.to_datetime(date_range[0])) & (filtered_df['date'] <= pd.to_datetime(date_range[1]))]

# Predict Current
dmatrix = xgb.DMatrix(filtered_df[required_features], feature_names=required_features)
filtered_df['predicted_demand'] = model.predict(dmatrix)

# Predict Future (next 30 days)
st.subheader("🔮 Future Demand Forecast (Next 30 Days)")
future_days = 30
last_date = df['date'].max()
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
latest_data = df.sort_values('date').groupby('product_id').tail(1)
future_rows = []

for date in future_dates:
    for _, row in latest_data.iterrows():
        future_rows.append({
            'store_id': row['store_id'],
            'product_id': row['product_id'],
            'price': row['price'],
            'stock_on_hand': row['stock_on_hand'],
            'promotion_flag': row['promotion_flag'],
            'day_of_week': date.weekday(),
            'day': date.day,
            'month': date.month,
            'is_weekend': int(date.weekday() in [5, 6]),
            'lag_1': row['units_sold'],
            'lag_7': row['units_sold'],
            'rolling_mean_7': row['units_sold'],
            'date': date
        })

future_df = pd.DataFrame(future_rows)
future_dmatrix = xgb.DMatrix(future_df[required_features], feature_names=required_features)
future_df['predicted_demand'] = model.predict(future_dmatrix)

fig_future, ax_future = plt.subplots(figsize=(12, 5))
sns.lineplot(data=future_df, x='date', y='predicted_demand', marker='o', ax=ax_future)
ax_future.set_title("Predicted Demand for Next 30 Days")
ax_future.set_ylabel("Predicted Units")
ax_future.set_xlabel("Date")
ax_future.tick_params(axis='x', rotation=45)
st.pyplot(fig_future)

st.download_button(
    label="📥 Download Future Predictions",
    data=future_df.to_csv(index=False),
    file_name="future_demand_predictions.csv",
    mime="text/csv"
)

st.subheader("📊 Predicted Demand Preview")
st.dataframe(filtered_df[['store_id', 'product_id', 'date', 'predicted_demand']].head(20))

st.subheader("📈 Demand Prediction vs Actuals")
if 'units_sold' in filtered_df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=filtered_df.head(100), x='date', y='units_sold', label='Actual', ax=ax)
    sns.lineplot(data=filtered_df.head(100), x='date', y='predicted_demand', label='Predicted', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Actual vs Predicted Demand")
    st.pyplot(fig)
else:
    st.warning("'units_sold' column not found in dataset. Skipping comparison plot.")

st.subheader("📌 Summary Statistics")
st.write(filtered_df['predicted_demand'].describe())

if {'price', 'purchase_cost'}.issubset(filtered_df.columns):
    filtered_df['predicted_profit'] = (filtered_df['price'] - filtered_df['purchase_cost']) * filtered_df['predicted_demand']
    if not filtered_df['predicted_profit'].empty and filtered_df['predicted_profit'].max() > 0:
        max_profit = filtered_df['predicted_profit'].max()
        max_profit_row = filtered_df.loc[filtered_df['predicted_profit'].idxmax()]

        st.subheader("💰 Predicted Profit Summary")
        st.metric(label="Total Predicted Profit", value=f"₹{filtered_df['predicted_profit'].sum():,.2f}")
        st.metric(label="Maximum Profit (Single Entry)", value=f"₹{max_profit:,.2f}")
        st.write("Details:")
        st.write(max_profit_row[['store_id', 'product_id', 'date', 'predicted_demand', 'price', 'purchase_cost', 'predicted_profit']])

        st.subheader("🏆 Top 10 Most Profitable Entries")
        top_10 = filtered_df.sort_values(by="predicted_profit", ascending=False).head(10)
        st.dataframe(top_10[['store_id', 'product_id', 'date', 'predicted_demand', 'price', 'purchase_cost', 'predicted_profit']])

        st.subheader("📉 Predicted Profit Over Time")
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=filtered_df.sort_values("date"), x="date", y="predicted_profit", ax=ax2)
        ax2.set_title("Predicted Profit Trend")
        ax2.set_ylabel("Profit (₹)")
        ax2.set_xlabel("Date")
        st.pyplot(fig2)
    else:
        st.warning("No profit data available to display summary and plots.")
else:
    st.warning("'price' and/or 'purchase_cost' columns missing. Cannot compute profit.")

st.download_button(
    label="📥 Download Filtered Predictions",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_predictions.csv",
    mime="text/csv"
)

if st.button("💾 Save All Predictions to CSV"):
    df['predicted_demand'] = model.predict(xgb.DMatrix(df[required_features], feature_names=required_features))
    df.to_csv("retail_dataset_with_predictions.csv", index=False)
    st.success("Saved as retail_dataset_with_predictions.csv")
