import streamlit as st


st.set_page_config(
    page_title="Financial News Sentiment & Market Stress Monitor",
    layout="wide",
)

st.title("Financial News Sentiment & Market Stress Monitor")
st.write(
    "This is a scaffold Streamlit app. You will connect it to derived datasets in "
    "`data/derived-data/` and add interactive charts (Altair / geopandas)."
)

with st.sidebar:
    st.header("Controls")
    st.selectbox("Bank / ETF", options=["KRE", "ZION", "PACW", "WAL"])
    st.date_input("Date range", value=())
    st.selectbox("Volatility metric", options=["Realized Volatility (RV)", "Absolute Return |r|"])
    st.selectbox("Sentiment smoothing window", options=["3-day", "7-day", "14-day"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("Time series (market + sentiment)")
    st.info("TODO: load derived data and render an interactive Altair chart.")

with col2:
    st.subheader("Sentiment vs volatility")
    st.info("TODO: render scatter plot with hover tooltips.")

st.subheader("Geographic stress map")
st.info("TODO: load shapefile + derived bank HQ mapping and render a choropleth.")

