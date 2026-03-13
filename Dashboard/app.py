
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page settings
st.set_page_config(page_title="Climate Analytics Dashboard", layout="wide")

st.title("🌍 Climate Trend & Extreme Weather Dashboard")

st.write("""
This dashboard analyzes climate data including **temperature, rainfall, humidity, and wind patterns**.
It also highlights **extreme climate events detected using machine learning anomaly detection**.
""")

df = pd.read_csv("C:\\Users\\avisw\\Desktop\\climate-trend-analysis-ml\\Data\\climate_features.csv")

# Sidebar filters
st.sidebar.header("Dashboard Controls")

variable = st.sidebar.selectbox(
    "Select Climate Variable",
    ["T2M","PRECTOTCORR","RH2M","WS10M"]
)

year_range = st.sidebar.slider(
    "Select Year Range",
    int(df["YEAR"].min()),
    int(df["YEAR"].max()),
    (int(df["YEAR"].min()), int(df["YEAR"].max()))
)

filtered = df[(df["YEAR"] >= year_range[0]) & (df["YEAR"] <= year_range[1])]

# Layout
col1, col2 = st.columns(2)

# Temperature trend
with col1:
    st.subheader("Temperature Trend")

    fig, ax = plt.subplots()
    ax.plot(filtered["T2M"], color="red")
    ax.set_title("Temperature Over Time")
    ax.set_ylabel("Temperature (°C)")
    st.pyplot(fig)

    st.write("""
    **Analysis:**  
    This plot shows long-term temperature fluctuations. Seasonal cycles and gradual warming
    patterns can often be observed in climate datasets.
    """)

# Rainfall trend
with col2:
    st.subheader("Rainfall Trend")

    fig, ax = plt.subplots()
    ax.plot(filtered["PRECTOTCORR"], color="blue")
    ax.set_title("Rainfall Over Time")
    ax.set_ylabel("Rainfall (mm)")
    st.pyplot(fig)

    st.write("""
    **Analysis:**  
    Rainfall patterns often show strong variability with occasional extreme peaks
    corresponding to intense storms or monsoon events.
    """)

# Distribution plots
st.subheader("Climate Variable Distributions")

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    sns.histplot(filtered["T2M"], bins=40, kde=True, color="orange")
    ax.set_title("Temperature Distribution")
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots()
    sns.histplot(filtered["PRECTOTCORR"], bins=40, kde=True, color="blue")
    ax.set_title("Rainfall Distribution")
    st.pyplot(fig)

st.write("""
**Insight:**  
Distribution plots help identify how frequently extreme climate values occur.
Rainfall distributions are usually highly skewed because most days have little rain,
while a few days experience heavy precipitation.
""")

# Correlation heatmap
st.subheader("Climate Variable Correlation")

corr = filtered[["T2M","RH2M","WS10M","PRECTOTCORR"]].corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm")
st.pyplot(fig)

st.write("""
**Interpretation:**  
Correlation analysis helps identify relationships between climate variables.
For example, humidity often correlates with rainfall events.
""")



# Summary statistics
st.subheader("Dataset Summary")

st.dataframe(filtered.describe())

st.write("""
**Summary:**  
This table provides statistical insights including mean temperature, rainfall variability,
and extreme values. Such metrics help quantify climate patterns across the dataset.
""")