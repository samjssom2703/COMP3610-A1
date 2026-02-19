import streamlit as st
import pandas as pd
import numpy as np
import os

CLEAN_PATH = os.path.join("data", "processed", "yellow_2024_01_clean.parquet")

st.set_page_config(page_title="Data Overview", layout="wide")

st.title("Data Overview")
st.markdown("Shows you the basics of the dataset used, so you know what you're working with before looking at the visualisations.")


@st.cache_data
def load_data():
    if "taxi_df" in st.session_state:
        return st.session_state["taxi_df"]
    if not os.path.exists(CLEAN_PATH):
        st.error("Dataset not ready yet. Open the Home page first to initialize data.")
        st.stop()
    df = pd.read_parquet(CLEAN_PATH)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    st.session_state["taxi_df"] = df
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
