import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Upload Data", layout="wide")

st.title("Upload Other Data")
st.markdown(
    """
    Got another dataset you want to poke around in?  
    Drop a CSV below and build interactive charts on the fly.
    """
)


uploaded_file = st.file_uploader(
    "Upload a CSV file:",
    type=["csv"],
    help="Maximum recommended size: 200 MB. Larger files may be slow.",
)

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()


@st.cache_data
def load_uploaded_csv(file):
    """Parse uploaded CSV. cache_data keyed on file content."""
    return pd.read_csv(file)


df = load_uploaded_csv(uploaded_file)

st.success(f"Loaded **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")


with st.expander("Data Preview", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)


with st.expander("Column Statistics"):
    st.dataframe(df.describe(include="all").T, use_container_width=True)


st.divider()
st.subheader("Build a Chart")
st.markdown("Choose your axes and chart type. We'll handle the rest.")

col1, col2, col3 = st.columns(3)

numeric_cols = df.select_dtypes(include="number").columns.tolist()
all_cols = df.columns.tolist()

with col1:
    x_col = st.selectbox("X-Axis", options=all_cols, index=0)

with col2:
    y_col = st.selectbox(
        "Y-Axis",
        options=numeric_cols if numeric_cols else all_cols,
        index=min(1, len(numeric_cols) - 1) if numeric_cols else 0,
    )

with col3:
    chart_type = st.selectbox(
        "Chart Type",
        options=["Bar", "Line", "Scatter", "Histogram", "Box"],
    )


color_col = st.selectbox(
    "Color / Group By (optional)",
    options=[None] + all_cols,
    index=0,
    format_func=lambda x: "None" if x is None else x,
)


try:
    if chart_type == "Bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
    elif chart_type == "Line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} over {x_col}")
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col}")
    elif chart_type == "Box":
        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
    else:
        fig = None

    if fig:
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Couldn't render chart: {e}")


st.divider()
st.subheader("Download")
st.download_button(
    label="Download data as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="uploaded_data.csv",
    mime="text/csv",
)
