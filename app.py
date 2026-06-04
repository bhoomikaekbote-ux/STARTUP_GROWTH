import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# --------------------------------
# Page Config
# --------------------------------

st.set_page_config(
    page_title="Startup Analytics Dashboard",
    page_icon="🚀",
    layout="wide"
)

# --------------------------------
# Load Dataset
# --------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("startup_data.csv")
    return df

df = load_data()

# --------------------------------
# Header
# --------------------------------

st.title("🚀 Startup Analytics Dashboard")
st.write("Interactive startup analytics with charts and insights")

# --------------------------------
# Sidebar Filters
# --------------------------------

st.sidebar.header("Filters")

industry = st.sidebar.multiselect(
    "Industry",
    options=df["Industry"].unique(),
    default=df["Industry"].unique()
)

region = st.sidebar.multiselect(
    "Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

filtered_df = df[
    (df["Industry"].isin(industry))
    &
    (df["Region"].isin(region))
]

# --------------------------------
# KPI Section
# --------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Startups",
    len(filtered_df)
)

c2.metric(
    "Funding",
    f"${filtered_df['Funding Amount (M USD)'].sum():.1f}M"
)

c3.metric(
    "Avg Revenue",
    f"${filtered_df['Revenue (M USD)'].mean():.1f}M"
)

c4.metric(
    "Avg Valuation",
    f"${filtered_df['Valuation (M USD)'].mean():.1f}M"
)

st.divider()

# --------------------------------
# Funding Analysis
# --------------------------------

st.subheader("Funding by Industry")

funding = filtered_df.groupby(
    "Industry"
)["Funding Amount (M USD)"].sum().reset_index()

fig1 = px.bar(
    funding,
    x="Industry",
    y="Funding Amount (M USD)",
    title="Funding Analysis"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# --------------------------------
# Revenue vs Valuation
# --------------------------------

st.subheader("Revenue vs Valuation")

fig2 = px.scatter(
    filtered_df,
    x="Revenue (M USD)",
    y="Valuation (M USD)",
    color="Industry",
    size="Employees",
    hover_name="Startup Name"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# --------------------------------
# Regional Analysis
# --------------------------------

st.subheader("Regional Distribution")

fig3 = px.pie(
    filtered_df,
    names="Region"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# --------------------------------
# Startup Trend
# --------------------------------

trend = filtered_df.groupby(
    "Year Founded"
).size().reset_index(
    name="Count"
)

fig4 = px.line(
    trend,
    x="Year Founded",
    y="Count",
    markers=True,
    title="Startup Growth Trend"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# --------------------------------
# AI Clustering
# --------------------------------

st.subheader("AI Startup Segmentation")

features = filtered_df[
[
"Funding Amount (M USD)",
"Revenue (M USD)",
"Valuation (M USD)"
]
]

if len(features) >= 3:

    model = KMeans(
        n_clusters=3,
        random_state=42
    )

    filtered_df["Cluster"] = model.fit_predict(
        features
    )

    fig5 = px.scatter(
        filtered_df,
        x="Funding Amount (M USD)",
        y="Valuation (M USD)",
        color=filtered_df[
            "Cluster"
        ].astype(str),
        hover_name="Startup Name"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

# --------------------------------
# Insights
# --------------------------------

st.subheader("Business Insights")

best_industry = filtered_df.groupby(
    "Industry"
)["Revenue (M USD)"].mean().idxmax()

best_region = filtered_df[
    "Region"
].mode()[0]

top_company = filtered_df.loc[
    filtered_df[
        "Valuation (M USD)"
    ].idxmax(),
    "Startup Name"
]

st.success(
    f"Highest revenue industry: {best_industry}"
)

st.info(
    f"Most active region: {best_region}"
)

st.warning(
    f"Highest valued startup: {top_company}"
)

# --------------------------------
# Data Table
# --------------------------------

st.subheader("Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)
