import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Netflix Content Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# HEADER (UI ONLY)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
.big-title {
    font-size: 42px;
    font-weight: 800;
}
.subtitle {
    font-size: 16px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎬 Netflix Content Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive dashboard to explore Netflix movies & TV shows</div>',
    unsafe_allow_html=True
)

st.markdown(
    "Upload a Netflix dataset (CSV or Excel). Expected columns: "
    "`type`, `title`, `director`, `cast`, `country`, `date_added`, "
    "`release_year`, `rating`, `duration`, `listed_in`."
)

# -----------------------------------------------------------------------------
# DATA CLEANING (UNCHANGED)
# -----------------------------------------------------------------------------
@st.cache_data
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if 'director' in df.columns:
        df['director'] = df['director'].fillna('Unknown')

    if 'date_added' in df.columns:
        df['date_added_parsed'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['added_year'] = df['date_added_parsed'].dt.year
        df['added_month'] = df['date_added_parsed'].dt.month
        df['added_month_name'] = df['date_added_parsed'].dt.month_name()
    else:
        df['date_added_parsed'] = pd.NaT
        df['added_year'] = pd.NA
        df['added_month'] = pd.NA
        df['added_month_name'] = pd.NA

    if 'type' not in df.columns:
        df['type'] = 'Movie'

    if 'listed_in' in df.columns:
        df['genres'] = df['listed_in'].fillna('').apply(
            lambda s: [g.strip() for g in s.split(',') if g.strip()]
        )
    else:
        df['genres'] = [[] for _ in range(len(df))]

    if 'director' in df.columns:
        df['director_popularity'] = df['director'].map(df['director'].value_counts())
    else:
        df['director_popularity'] = 0

    return df

# -----------------------------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------------------------
uploaded = st.file_uploader("📂 Upload Netflix CSV or Excel file", type=["csv", "xlsx"])

if uploaded is None:
    st.info("⬆ Upload a Netflix dataset to begin analysis")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded, engine="openpyxl")
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

st.success("✅ File loaded and cleaned successfully")
df = clean_dataframe(df)

# -----------------------------------------------------------------------------
# SIDEBAR FILTERS (UI ONLY)
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🔎 Filters")

years = sorted(pd.Series(df['added_year'].dropna().unique()).astype(int).tolist())
min_year, max_year = (min(years), max(years)) if years else (None, None)

with st.sidebar.expander("📅 Time Range", expanded=True):
    if years:
        year_range = st.slider(
            "Added Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (None, None)

types = ["All"] + sorted(df["type"].dropna().unique().tolist())
with st.sidebar.expander("🎞 Content Type"):
    sel_type = st.selectbox("Type", types)

all_genres = sorted({g for sub in df["genres"] for g in sub})
with st.sidebar.expander("🎭 Genres"):
    sel_genres = st.multiselect(
        "Select Genres",
        options=all_genres,
        default=all_genres[:3]
    )

top_directors = df["director"].value_counts().head(10).index.tolist()
with st.sidebar.expander("🎬 Director"):
    sel_director = st.selectbox("Top Directors", ["All"] + top_directors)

# -----------------------------------------------------------------------------
# APPLY FILTERS (UNCHANGED LOGIC)
# -----------------------------------------------------------------------------
df_filtered = df.copy()

if year_range[0] is not None:
    df_filtered = df_filtered[
        df_filtered["added_year"].between(year_range[0], year_range[1], inclusive="both")
    ]

if sel_type != "All":
    df_filtered = df_filtered[df_filtered["type"] == sel_type]

if sel_genres:
    df_filtered = df_filtered[
        df_filtered["genres"].apply(lambda g: any(x in g for x in sel_genres))
    ]

if sel_director != "All":
    df_filtered = df_filtered[df_filtered["director"] == sel_director]

# -----------------------------------------------------------------------------
# TABS (MAJOR UI IMPROVEMENT)
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Trends",
    "🎭 Genres & Directors",
    "📄 Data"
])

# -----------------------------------------------------------------------------
# TAB 1 — OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    total_titles = len(df_filtered)
    movies_count = len(df_filtered[df_filtered["type"] == "Movie"])
    tv_count = len(df_filtered[df_filtered["type"] == "TV Show"])

    avg_per_year = None
    if not df_filtered["added_year"].dropna().empty:
        yrs = df_filtered["added_year"].dropna().astype(int)
        avg_per_year = int(len(df_filtered) / (yrs.max() - yrs.min() + 1))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🎞 Total Titles", f"{total_titles:,}")
    c2.metric("🎬 Movies", f"{movies_count:,}")
    c3.metric("📺 TV Shows", f"{tv_count:,}")
    c4.metric("📈 Avg / Year", f"{avg_per_year:,}" if avg_per_year else "N/A")

    spark = df_filtered.dropna(subset=["added_year"]).groupby("added_year").size().reset_index(name="count")
    if not spark.empty:
        fig = px.area(spark, x="added_year", y="count", height=200)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2 — TRENDS
# -----------------------------------------------------------------------------
with tab2:
    df_year = df_filtered.dropna(subset=["added_year"]).groupby(
        ["added_year", "type"]
    ).size().reset_index(name="count")

    fig1 = px.line(
        df_year,
        x="added_year",
        y="count",
        color="type",
        markers=True,
        title="Movies vs TV Shows Added Per Year"
    )
    st.plotly_chart(fig1, use_container_width=True)

    heat = df_filtered.dropna(subset=["added_year", "added_month"]).groupby(
        ["added_year", "added_month"]
    ).size().reset_index(name="count")

    heat_pivot = heat.pivot(index="added_month", columns="added_year", values="count").fillna(0)
    fig2 = go.Figure(
        data=go.Heatmap(
            z=heat_pivot.values,
            x=heat_pivot.columns.astype(str),
            y=[datetime(2000, int(m), 1).strftime("%b") for m in heat_pivot.index],
            colorscale="Blues"
        )
    )
    fig2.update_layout(title="Monthly Additions Heatmap")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3 — GENRES & DIRECTORS
# -----------------------------------------------------------------------------
with tab3:
    top_dir = df_filtered["director"].value_counts().head(20).reset_index()
    top_dir.columns = ["director", "count"]
    fig_dir = px.bar(
        top_dir,
        x="count",
        y="director",
        orientation="h",
        title="Top Directors"
    )
    st.plotly_chart(fig_dir, use_container_width=True)

    with st.expander("📌 Advanced Analytics"):
        mins = df_filtered["duration"].astype(str).str.extract(r"(\d+)").astype(float)
        if not mins.dropna().empty:
            fig_dur = px.histogram(mins[0], nbins=40, title="Duration Distribution (Minutes)")
            st.plotly_chart(fig_dur, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 4 — DATA
# -----------------------------------------------------------------------------
with tab4:
    st.dataframe(df_filtered.reset_index(drop=True))
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download Filtered CSV",
        csv,
        "netflix_filtered.csv",
        "text/csv"
    )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
---
<center style="color:gray">
Built with ❤️ using Streamlit | Netflix Analytics Dashboard
</center>
""", unsafe_allow_html=True)
