import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MovieVerse Recommender",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_movies():
    with open("movies.pkl", "rb") as f:
        df = pickle.load(f)

    # NORMALISASI NAMA KOLOM
    df.columns = df.columns.str.lower()

    # DETEKSI KOLOM DESKRIPSI
    text_column = None
    for col in ["overview", "description", "plot", "summary"]:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise ValueError(
            "Tidak ditemukan kolom teks film (overview/description/plot)"
        )

    df[text_column] = df[text_column].fillna("")
    df = df.reset_index(drop=True)

    return df, text_column

movies, text_col = load_movies()

# =========================
# TF-IDF & SIMILARITY
# =========================
@st.cache_resource
def build_similarity(data, column):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = tfidf.fit_transform(data[column])
    return cosine_similarity(matrix)

similarity_matrix = build_similarity(movies, text_col)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend(title, n=5):
    if title not in movies["title"].values:
        return []

    idx = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    return movies.iloc[[i[0] for i in scores]]["title"].tolist()

# =========================
# UI
# =========================
st.title("🎬 MovieVerse Recommender")
st.caption("Content-Based Filtering | TF-IDF + Cosine Similarity")

selected_movie = st.selectbox(
    "Pilih Film",
    movies["title"].values
)

num_recommend = st.slider("Jumlah Rekomendasi", 3, 10, 5)

if selected_movie:
    movie_data = movies[movies["title"] == selected_movie].iloc[0]

    st.subheader("🎥 Film Dipilih")
    st.write(movie_data[text_col])

    st.subheader("✨ Rekomendasi Film Serupa")

    recs = recommend(selected_movie, num_recommend)

    for r in recs:
        st.markdown(f"- 🎬 **{r}**")
