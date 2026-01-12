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
    df["overview"] = df["overview"].fillna("")
    return df.reset_index(drop=True)

movies = load_movies()

# =========================
# TF-IDF & SIMILARITY
# =========================
@st.cache_resource
def build_similarity(data):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(data["overview"])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

similarity_matrix = build_similarity(movies)

# =========================
# RECOMMENDATION FUNCTION
# =========================
def recommend(movie_title, n=5):
    if movie_title not in movies["title"].values:
        return []

    idx = movies[movies["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    recommendations = [movies.iloc[i[0]]["title"] for i in scores]
    return recommendations

# =========================
# UI HEADER
# =========================
st.title("🎬 MovieVerse Recommender")
st.caption("Content-Based Movie Recommendation System")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_movie = st.selectbox(
        "Pilih Film Favorit",
        movies["title"].values
    )
    num_recommend = st.slider(
        "Jumlah Rekomendasi",
        3, 10, 5
    )

# =========================
# MAIN CONTENT
# =========================
if selected_movie:
    st.subheader("🎥 Film yang Dipilih")

    movie_data = movies[movies["title"] == selected_movie].iloc[0]

    st.markdown(f"""
    <div style="background:#020617;padding:20px;border-radius:15px">
        <h3>{movie_data['title']}</h3>
        <p style="color:#94a3b8">{movie_data['overview']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("✨ Rekomendasi Film Serupa")

    recommendations = recommend(selected_movie, num_recommend)

    if recommendations:
        cols = st.columns(3)
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:#020617;padding:15px;border-radius:12px">
                    <h4>{rec}</h4>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Rekomendasi tidak ditemukan.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("🚀 Built with Streamlit | Content-Based Filtering")
