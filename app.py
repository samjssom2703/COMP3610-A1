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
CLEAN_PATH = os.path.join(PROCESSED_DIR, "yellow_2024_01_clean.parquet")

st.set_page_config(
    page_title="NYC Taxi Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4A90D9;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)


def _download(url, dest):
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)


def _build_clean_dataset():
    import gc
    import pyarrow.parquet as pq

    with st.spinner("First run — downloading raw data (~50 MB)…"):
        _download(TRIP_URL, TRIP_RAW)
        _download(ZONE_URL, ZONE_CSV)
    with st.spinner("Cleaning & engineering features — this may take a minute…"):
        # Read only needed columns to save memory
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
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "payment_type",
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
        df = df.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime","PULocationID","DOLocationID","fare_amount"])
        df = df[df["tpep_dropoff_datetime"]>df["tpep_pickup_datetime"]]
        df = df[df["trip_distance"]>0]
        df = df[(df["fare_amount"]>0)&(df["fare_amount"]<=500)]
        df = df[df["total_amount"]>0]
        df["trip_duration_minutes"] = (df["tpep_dropoff_datetime"]-df["tpep_pickup_datetime"]).dt.total_seconds()/60
        df = df[(df["trip_distance"]<=200)&(df["trip_duration_minutes"]>=1)&(df["trip_duration_minutes"]<=300)]
        df = df[(df["passenger_count"]>=1)&(df["passenger_count"]<=9)]
        df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
        df["pickup_day_of_week"] = df["tpep_pickup_datetime"].dt.day_name()
        df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
        df["trip_speed_mph"] = np.where(df["trip_duration_minutes"]>0, df["trip_distance"]/(df["trip_duration_minutes"]/60), 0)
        df.loc[df["trip_speed_mph"]>80,"trip_speed_mph"] = np.nan
        df["tip_pct"] = np.where(df["fare_amount"]>0, (df["tip_amount"]/df["fare_amount"])*100, 0)
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


@st.cache_data
def load_zones():
    if not os.path.exists(ZONE_CSV):
        _download(ZONE_URL, ZONE_CSV)
    return pd.read_csv(ZONE_CSV)


df = load_data()
zone_df = load_zones()

if "taxi_df" not in st.session_state:
    st.session_state["taxi_df"] = df
if "zone_df" not in st.session_state:
    st.session_state["zone_df"] = zone_df


st.markdown('<p class="main-header">NYC Yellow Taxi Trip Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Exploring Yellow Taxi Data from January 2024</p>', unsafe_allow_html=True)
st.markdown(
    """
    This dashboard explores **NYC Yellow Taxi trip records for January 2024** (~3 million trips).
    The data has been cleaned, validated, and enriched with derived features such as trip duration,
    speed, pickup hour, and day of week. Use the **sidebar** to navigate to the Visualizations
    page and apply interactive filters to all charts.
    """
)

st.divider()
st.subheader("Key Metrics at a Glance")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric(
        label="Total Trips",
        value=f"{len(df):,}",
        help="Total number of cleaned taxi trips in January 2024",
    )

with col2:
    avg_fare = df["fare_amount"].mean()
    st.metric(
        label="Average Fare",
        value=f"${avg_fare:.2f}",
        help="Mean fare amount across all trips",
    )

with col3:
    total_rev = df["total_amount"].sum()
    st.metric(
        label="Total Revenue",
        value=f"${total_rev:,.0f}",
        help="Sum of total_amount for all trips",
    )

with col4:
    avg_dist = df["trip_distance"].mean()
    st.metric(
        label="Avg. Trip Distance",
        value=f"{avg_dist:.2f} mi",
        help="Most NYC taxi trips are pretty short, actually",
    )

with col5:
    avg_dur = df["trip_duration_minutes"].mean()
    st.metric(
        label="Avg. Trip Duration",
        value=f"{avg_dur:.1f} min",
        help="Includes time stuck in traffic, of course",
    )

st.divider()
st.subheader("Data Coverage")

c1, c2, c3 = st.columns(3)
with c1:
    min_date = df["pickup_date"].min()
    max_date = df["pickup_date"].max()
    st.info(f"**Date Range:** {min_date} to {max_date}")

with c2:
    st.info(f"**Rows After Cleaning:** {len(df):,}")

with c3:
    payment_map = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute"}
    top_pay = df["payment_type"].map(payment_map).value_counts().idxmax()
    st.info(f"**Most Common Payment:** {top_pay}")

st.divider()
st.subheader("About This Dashboard")

st.markdown(
    """
    Use the **sidebar** to navigate between pages:

    | Page | What's There |
    |------|-------------|
    | **Overview** | Data quality info, column descriptions, basic stats |
    | **Visualisations** | All 5 interactive Plotly charts with filters — the fun stuff |
    | **Upload Data** | Bring your own CSV and make charts |

    **Filters available on the Visualizations page:**
    - Date range selector
    - Hour of day range slider (0–23)
    - Payment type multi-select dropdown

    All visualisations update dynamically when filters change.

    Built with Streamlit, Plotly, and DuckDB for COMP 3610 Assignment 1.
    """
)


st.sidebar.success("Select a page above to explore!")
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset:** NYC Yellow Taxi (Jan 2024)")
st.sidebar.markdown(f"**Total Trips:** {len(df):,}")
st.sidebar.markdown(f"**Revenue:** ${df['total_amount'].sum():,.0f}")