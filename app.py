import streamlit as st
import pandas as pd
import numpy as np
import os

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


@st.cache_data
def load_data():
    clean_path = os.path.join("data", "processed", "yellow_2024_01_clean.parquet")
    if not os.path.exists(clean_path):
        st.error(
            "Cleaned dataset not found! "
            "Please run assignment1.ipynb first to generate data/processed/yellow_2024_01_clean.parquet"
        )
        st.stop()

    df = pd.read_parquet(clean_path)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    return df


@st.cache_data
def load_zones():
    """Load the taxi zone lookup CSV for zone name resolution."""
    zone_path = os.path.join("data", "raw", "taxi_zone_lookup.csv")
    if not os.path.exists(zone_path):
        st.error("Zone lookup CSV not found. Please run assignment1.ipynb first.")
        st.stop()
    return pd.read_csv(zone_path)


df = load_data()
zone_df = load_zones()


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
    avg_dur = df["trip_duration_min"].mean()
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