import streamlit as st
import pandas as pd
import numpy as np
import requests
import os

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
TRIP_RAW = os.path.join(RAW_DIR, "yellow_tripdata_2024-01.parquet")
ZONE_CSV = os.path.join(RAW_DIR, "taxi_zone_lookup.csv")
CLEAN_PATH = os.path.join("data", "processed", "yellow_2024_01_clean.parquet")

st.set_page_config(page_title="Data Overview", layout="wide")

st.title("Data Overview")
st.markdown("Shows you the basics of the dataset used, so you know what you're working with before looking at the visualisations.")


def _download(url, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as file:
        for chunk in response.iter_content(8192):
            file.write(chunk)


def _build_clean_dataset():
    import gc
    import pyarrow.parquet as pq

    with st.spinner("Preparing dataset for first use…"):
        _download(TRIP_URL, TRIP_RAW)
        _download(ZONE_URL, ZONE_CSV)

        needed = [
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "PULocationID", "DOLocationID", "passenger_count",
            "trip_distance", "fare_amount", "tip_amount",
            "total_amount", "payment_type", "VendorID", "RatecodeID",
            "store_and_fwd_flag", "extra", "mta_tax", "tolls_amount",
            "improvement_surcharge", "congestion_surcharge", "airport_fee",
        ]
        available_cols = set(pq.ParquetFile(TRIP_RAW).schema_arrow.names)
        selected_cols = [column for column in needed if column in available_cols]

        required_cols = {
            "tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID",
            "passenger_count", "trip_distance", "fare_amount", "tip_amount", "total_amount", "payment_type",
        }
        missing_required = sorted(required_cols - available_cols)
        if missing_required:
            raise RuntimeError(f"Required columns missing in source parquet: {missing_required}")

        table = pq.read_table(TRIP_RAW, columns=selected_cols)
        df = table.to_pandas()
        del table
        gc.collect()

        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        df = df[(df["tpep_pickup_datetime"] >= "2024-01-01") & (df["tpep_pickup_datetime"] < "2024-02-01")]
        df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID", "DOLocationID", "fare_amount"])
        df = df[df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"]]
        df = df[df["trip_distance"] > 0]
        df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 500)]
        df = df[df["total_amount"] > 0]
        df["trip_duration_minutes"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
        df = df[(df["trip_distance"] <= 200) & (df["trip_duration_minutes"] >= 1) & (df["trip_duration_minutes"] <= 300)]
        df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= 9)]
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
        df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
        df["trip_speed_mph"] = np.where(df["trip_duration_minutes"] > 0, df["trip_distance"] / (df["trip_duration_minutes"] / 60), 0)
        df.loc[df["trip_speed_mph"] > 80, "trip_speed_mph"] = np.nan
        df["tip_pct"] = np.where(df["fare_amount"] > 0, (df["tip_amount"] / df["fare_amount"]) * 100, 0)

        os.makedirs(PROCESSED_DIR, exist_ok=True)
        df.to_parquet(CLEAN_PATH, index=False)
        del df
        gc.collect()


@st.cache_data
def load_data():
    if not os.path.exists(CLEAN_PATH):
        _build_clean_dataset()
    df = pd.read_parquet(CLEAN_PATH)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    return df


df = load_data()


st.subheader("Dataset at a Glance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", f"{len(df.columns)}")
col3.metric(
    "Date Range",
    f"{(df['pickup_date'].max() - df['pickup_date'].min()).days + 1} days",
)
col4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


st.divider()
tab1, tab2, tab3 = st.tabs(["Statistics", "Data Sample", "Column Info"])

with tab1:
    st.subheader("Summary Statistics")
    st.markdown("The classic `.describe()` output, but make it interactive.")

    # Let users pick which columns they care about
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        "Pick your columns:",
        numeric_cols,
        default=[
            c
            for c in [
                "fare_amount",
                "trip_distance",
                "tip_amount",
                "trip_duration_minutes",
                "trip_speed_mph",
            ]
            if c in numeric_cols
        ],
    )

    if selected_cols:
        stats_df = df[selected_cols].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

        # Why not let them download it?
        csv = stats_df.to_csv()
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="taxi_statistics.csv",
            mime="text/csv",
        )

with tab2:
    st.subheader("Peek at the Data")

    # Slider for how many rows to show — simple but useful
    num_rows = st.slider(
        "How many rows?", min_value=5, max_value=100, value=20, step=5
    )

    # And which columns?
    all_columns = df.columns.tolist()
    display_cols = st.multiselect(
        "Which columns to show:",
        all_columns,
        default=[
            c
            for c in [
                "tpep_pickup_datetime",
                "fare_amount",
                "trip_distance",
                "passenger_count",
                "payment_type",
                "tip_amount",
                "pickup_hour",
                "pickup_day_of_week",
            ]
            if c in all_columns
        ],
    )

    if display_cols:
        st.dataframe(
            df[display_cols].head(num_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("You gotta pick at least one column!")

with tab3:
    st.subheader("What's in Each Column?")

    # Handy reference for what all these column names mean
    column_info = {
        "VendorID": "Which taxi company (1=CMT, 2=VTS)",
        "tpep_pickup_datetime": "When the trip started",
        "tpep_dropoff_datetime": "When the trip ended",
        "passenger_count": "How many people (driver enters this, so... grain of salt)",
        "trip_distance": "Miles traveled according to the meter",
        "RatecodeID": "Rate type (1=Standard, 2=JFK, 3=Newark, etc.)",
        "store_and_fwd_flag": "Was trip data stored before sending? (Y/N)",
        "PULocationID": "Pickup zone ID (there are 263 zones in NYC)",
        "DOLocationID": "Dropoff zone ID",
        "payment_type": "1=Card, 2=Cash, 3=No charge, 4=Dispute",
        "fare_amount": "The meter fare (before tips/tolls/extras)",
        "extra": "Rush hour and overnight surcharges",
        "mta_tax": "MTA tax — always $0.50",
        "tip_amount": "Tip (only recorded for card payments!)",
        "tolls_amount": "Bridge/tunnel tolls",
        "improvement_surcharge": "$0.30 for taxi improvements",
        "total_amount": "Everything added up",
        "congestion_surcharge": "Manhattan congestion fee",
        "airport_fee": "For airport pickups",
        "trip_duration_minutes": "[Derived] Trip length in minutes",
        "trip_speed_mph": "[Derived] Distance ÷ duration in hours",
        "pickup_hour": "[Derived] Hour of pickup (0–23)",
        "pickup_day_of_week": "[Derived] Day name (Monday–Sunday)",
        "pickup_date": "[Derived] Just the date part",
    }

    # Table built at this point 
    info_data = []
    for col in df.columns:
        info_data.append(
            {
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": f"{df[col].notna().sum():,}",
                "Null %": f"{df[col].isna().mean() * 100:.1f}%",
                "Description": column_info.get(col, "—"),
            }
        )

    info_df = pd.DataFrame(info_data)

    with st.expander("Click to see all columns", expanded=True):
        st.dataframe(info_df, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Data Quality Check")
st.caption("Because garbage in = garbage out")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Missing Values:**")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame(
        {
            "Column": missing.index,
            "Missing": missing.values,
            "Percentage": missing_pct.values,
        }
    )
    missing_df = missing_df[missing_df["Missing"] > 0].sort_values(
        "Missing", ascending=False
    )

    if len(missing_df) > 0:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("No missing values in critical columns! The cleaning pipeline did its job.")

with col2:
    st.markdown("**Value Ranges:**")
    st.caption("Just to make sure nothing crazy slipped through our filters")
    ranges_data = {
        "Column": [
            "fare_amount",
            "trip_distance",
            "tip_amount",
            "trip_duration_minutes",
            "trip_speed_mph",
        ],
        "Min": [
            f"${df['fare_amount'].min():.2f}",
            f"{df['trip_distance'].min():.2f} mi",
            f"${df['tip_amount'].min():.2f}",
            f"{df['trip_duration_minutes'].min():.1f} min",
            f"{df['trip_speed_mph'].min():.1f} mph",
        ],
        "Max": [
            f"${df['fare_amount'].max():.2f}",
            f"{df['trip_distance'].max():.2f} mi",
            f"${df['tip_amount'].max():.2f}",
            f"{df['trip_duration_minutes'].max():.1f} min",
            f"{df['trip_speed_mph'].max():.1f} mph",
        ],
        "Mean": [
            f"${df['fare_amount'].mean():.2f}",
            f"{df['trip_distance'].mean():.2f} mi",
            f"${df['tip_amount'].mean():.2f}",
            f"{df['trip_duration_minutes'].mean():.1f} min",
            f"{df['trip_speed_mph'].mean():.1f} mph",
        ],
    }
    st.dataframe(pd.DataFrame(ranges_data), use_container_width=True, hide_index=True)
