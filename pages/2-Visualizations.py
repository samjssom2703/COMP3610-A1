import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import os

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
TRIP_RAW = os.path.join(RAW_DIR, "yellow_tripdata_2024-01.parquet")
ZONE_CSV = os.path.join(RAW_DIR, "taxi_zone_lookup.csv")
CLEAN_PATH = os.path.join(PROCESSED_DIR, "yellow_2024_01_clean.parquet")

st.set_page_config(page_title="Visualisations", layout="wide")

st.title("Visualisations Of The Data")
st.markdown("Use the filters in the sidebar to segment different data by date, hour, and payment type. Each chart will update accordingly, so you can spot patterns across different slices of the data.")


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
    with st.spinner("First run — downloading raw data (~50 MB)…"):
        _download(TRIP_URL, TRIP_RAW)
        _download(ZONE_URL, ZONE_CSV)
    with st.spinner("Cleaning & engineering features — this may take a minute…"):
        df = pd.read_parquet(TRIP_RAW)
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
        df = df.dropna(subset=["tpep_pickup_datetime","tpep_dropoff_datetime","PULocationID","DOLocationID","fare_amount"])
        df = df[(df["tpep_pickup_datetime"]>="2024-01-01")&(df["tpep_pickup_datetime"]<"2024-02-01")]
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


st.sidebar.header("Filters")

# Date range selector - can be used to zoom into specific weeks or single days
st.sidebar.subheader("Date Range")
min_date = df["pickup_date"].min()
max_date = df["pickup_date"].max()

date_range = st.sidebar.date_input(
    "Pick your dates:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# An exception - handling case if user only selects one date
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range


st.sidebar.subheader("Hour of Day")
hour_range = st.sidebar.slider(
    "Select hour range:",
    min_value=0,
    max_value=23,
    value=(0, 23),
    step=1,
)


st.sidebar.subheader("Payment Type")
PAYMENT_MAP = {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute", 5: "Unknown"}
df["payment_name"] = df["payment_type"].map(PAYMENT_MAP)
all_payment_types = sorted(df["payment_name"].dropna().unique().tolist())

selected_payments = st.sidebar.multiselect(
    "Select payment types:",
    options=all_payment_types,
    default=all_payment_types,
)


filtered = df[
    (df["pickup_date"] >= start_date)
    & (df["pickup_date"] <= end_date)
    & (df["pickup_hour"] >= hour_range[0])
    & (df["pickup_hour"] <= hour_range[1])
    & (df["payment_name"].isin(selected_payments))
]


st.sidebar.divider()
st.sidebar.metric("Filtered Trips", f"{len(filtered):,}")
st.sidebar.caption(f"out of {len(df):,} total ({len(filtered)/len(df)*100:.1f}%)")

if len(filtered) == 0:
    st.warning("No trips match those filters. Try loosening up a bit?")
    st.stop()


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top Pickup Zones",
    "Avg Fare by Hour",
    "Trip Distance",
    "Payment Types",
    "(Day x Hour) Heatmap",
])

# Bar Chart containing Top 10 Pickup Zones
with tab1:
    st.subheader("Top 10 Pickup Zones by Trip Count")

    pickup_counts = (
        filtered.groupby("PULocationID")
        .size()
        .reset_index(name="trip_count")
        .merge(
            zone_df[["LocationID", "Zone"]],
            left_on="PULocationID",
            right_on="LocationID",
        )
        .nlargest(10, "trip_count")
        .sort_values("trip_count", ascending=True)
    )

    fig1 = px.bar(
        pickup_counts,
        x="trip_count",
        y="Zone",
        orientation="h",
        title="Top 10 Pickup Zones by Trip Count",
        labels={"trip_count": "Number of Trips", "Zone": "Pickup Zone"},
        color="trip_count",
        color_continuous_scale="Blues",
    )
    fig1.update_layout(height=500, yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        """
        **Insight:** Manhattan dominates taxi pickups, with zones like Upper East Side South,
        Midtown Center, and Penn Station/Madison Square West consistently ranking at the top.
        These are major commercial, transit, and tourist hubs. Outer-borough zones rarely
        appear in the top 10, confirming that taxi demand is heavily concentrated in central Manhattan.
        """
    )

# Line Chart showing Average Fare by Hour
with tab2:
    st.subheader("Average Fare by Hour of Day")
    st.caption("Reveals hourly pricing patterns across the day")

    hourly_fare = (
        filtered.groupby("pickup_hour")["fare_amount"]
        .mean()
        .reset_index()
        .rename(columns={"fare_amount": "avg_fare"})
    )

    fig2 = px.line(
        hourly_fare,
        x="pickup_hour",
        y="avg_fare",
        title="Average Fare by Hour of Day",
        labels={"pickup_hour": "Hour of Day", "avg_fare": "Average Fare ($)"},
        markers=True,
    )
    fig2.update_layout(height=400)
    fig2.update_xaxes(tickmode="linear", tick0=0, dtick=1)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        """
        **Insight:** Average fares tend to peak during the early morning hours (4–6 AM),
        likely because trips at that time are longer-distance airport or commuter runs when
        traffic is light. Midday fares are relatively stable, while a slight uptick in the
        late evening reflects nighttime surcharges and longer cross-town trips.
        """
    )

# Histogram showing Trip Distance Distribution
with tab3:
    st.subheader("Distribution of Trip Distances")
    st.caption("How far do people actually go in a NYC taxi?")

    # filter to distances <= 30 miles 
    dist_df = filtered[filtered["trip_distance"] <= 30]

    fig3 = px.histogram(
        dist_df,
        x="trip_distance",
        nbins=60,
        title="Distribution of Trip Distances",
        labels={"trip_distance": "Trip Distance (miles)", "count": "Number of Trips"},
        color_discrete_sequence=["#636EFA"],
    )
    fig3.add_vline(
        x=dist_df["trip_distance"].median(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {dist_df['trip_distance'].median():.2f} mi",
    )
    fig3.update_layout(height=450)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        """
        **Insight:** The distribution is heavily right-skewed, with the vast majority of trips
        under 5 miles. The median sits around 1.5–2 miles, confirming that most NYC taxi rides
        are short hops within Manhattan. The long tail extending past 15 miles represents
        airport trips (JFK, LaGuardia) and outer-borough journeys.
        """
    )

# Pie Chart
with tab4:
    st.subheader("Payment Type Breakdown")

    payment_counts = (
        filtered["payment_name"]
        .value_counts()
        .reset_index()
    )
    payment_counts.columns = ["Payment Type", "Trip Count"]

    fig4 = px.pie(
        payment_counts,
        values="Trip Count",
        names="Payment Type",
        title="Breakdown of Payment Types",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig4.update_traces(textposition="inside", textinfo="percent+label")
    fig4.update_layout(height=450)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        """
        **Insight:** Credit card payments overwhelmingly dominate, accounting for roughly 90%+
        of all trips. Cash usage has declined significantly in recent years as contactless payment
        has become the norm. The small "No Charge" and "Dispute" slices represent comped rides
        and fare disputes — negligible in volume but present in the data.
        """
    )

# Heatmap which shows Trips by Day of Week × Hour
with tab5:
    st.subheader("Trip Volume by Day of Week and Hour")
    st.caption("Spot the patterns - when are taxis busiest?")

    DAY_ORDER = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]

    heatmap_data = (
        filtered.groupby(["pickup_day_of_week", "pickup_hour"])
        .size()
        .reset_index(name="trip_count")
        .pivot(index="pickup_day_of_week", columns="pickup_hour", values="trip_count")
        .reindex(index=DAY_ORDER, columns=range(24), fill_value=0)
    )

    fig5 = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Trip Count"),
        x=list(range(24)),
        y=DAY_ORDER,
        color_continuous_scale="YlOrRd",
        title="Trip Volume by Day of Week and Hour",
    )
    fig5.update_layout(height=500)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown(
        """
        **Insight:** Weekday patterns show two clear peaks: a morning rush around 8–9 AM
        and a stronger evening peak from 5–7 PM, reflecting commuter behaviour. Weekends
        shift dramatically — Saturday and Sunday mornings are quiet, but late-night activity
        (11 PM–2 AM) is significantly higher than on weekdays, indicating nightlife-driven demand.
        """
    )

    st.info(
        "**Pro tip:** Use the hour slider in the sidebar to zoom into rush hour "
        "(e.g., 7–9 AM) or late night (10 PM–2 AM), you'll see that all charts update!"
    )
