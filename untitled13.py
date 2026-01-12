import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MovieVerse",
    page_icon="🎬",
    layout="wide"
)

# =========================
# MODERN CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #0b0f19;
}
.main {
    background-color: #0b0f19;
}
h1, h2, h3, h4 {
    color: #f9fafb;
}
p, label {
    color: #cbd5e1;
}
.card {
    background: linear-gradient(145deg, #020617, #020617);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0px 15px 30px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: white;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    margin-bottom: 10px;
}
.rec-card {
    background: #020617;
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    transition: 0.3s;
}
.rec-card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 12px 25px rgba(99,102,241,0.4);
}
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 60px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_movies():
    with open("movies.pkl", "rb") as f:
        df = pickle.load(f)

    df.columns = df.columns.str.lower()

    text_col = None
    for col in ["overview", "description", "plot", "summary"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        raise ValueError("Kolom deskripsi film tidak ditemukan")

    df[text_col] = df[text_col].fillna("")
    return df.reset_index(drop=True), text_col

movies, text_col = load_movies()

# =========================
# TF-IDF SIMILARITY
# =========================
@st.cache_resource
def build_similarity(data, column):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(data[column])
    return cosine_similarity(matrix)

similarity_matrix = build_similarity(movies, text_col)

# =========================
# RECOMMEND FUNCTION
# =========================
def recommend(title, n=6):
    idx = movies[movies["title"] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return movies.iloc[[i[0] for i in scores]]

# =========================
# HEADER
# =========================
st.markdown("""
<h1>🎬 MovieVerse</h1>
<p>Discover movies through intelligent content-based recommendations</p>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    selected_movie = st.selectbox(
        "🎥 Pilih Film Favorit",
        movies["title"].values
    )
    top_n = st.slider("Jumlah Rekomendasi", 3, 12, 6)

# =========================
# MAIN CONTENT
# =========================
if selected_movie:
    movie_data = movies[movies["title"] == selected_movie].iloc[0]

    # SELECTED MOVIE CARD
    st.markdown(f"""
    <div class="card">
        <span class="badge">FILM DIPILIH</span>
        <h2>{movie_data['title']}</h2>
        <p>{movie_data[text_col][:500]}...</p>
    </div>
    """, unsafe_allow_html=True)

    # RECOMMENDATIONS
    st.subheader("✨ Rekomendasi Film Serupa")

    recommendations = recommend(selected_movie, top_n)

    cols = st.columns(3)
    for i, row in recommendations.iterrows():
        with cols[i % 3]:
            st.markdown(f"""
            <div class="rec-card">
                <h4>{row['title']}</h4>
            </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
🚀 MovieVerse • Content-Based Recommender System  
<br>TF-IDF & Cosine Similarity
</div>
""", unsafe_allow_html=True)
