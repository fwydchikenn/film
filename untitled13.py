import streamlit as st
import pandas as pd
import pickle
import numpy as np
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
# LIGHT MODERN CSS
# =========================
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.main {
    background-color: #f8fafc;
}
h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
}
.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0px 8px 24px rgba(15, 23, 42, 0.08);
}
.rec-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0px 6px 18px rgba(15, 23, 42, 0.08);
    transition: 0.3s ease;
}
.rec-card:hover {
    transform: translateY(-6px);
    box-shadow: 0px 14px 30px rgba(37, 99, 235, 0.25);
}
.badge {
    background: linear-gradient(135deg, #2563eb, #38bdf8);
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 12px;
    color: white;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 60px;
    font-size: 13px;
}
.stMultiSelect label {
    font-weight: 600;
    color: #0f172a;
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

    text_col = next(
        (c for c in ["overview", "description", "plot", "summary"] if c in df.columns),
        None
    )

    if text_col is None:
        raise ValueError("Kolom deskripsi film tidak ditemukan")

    df[text_col] = df[text_col].fillna("")
    return df.reset_index(drop=True), text_col

movies, text_col = load_movies()

# =========================
# TF-IDF MODEL
# =========================
@st.cache_resource
def build_model(data, column):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    tfidf_matrix = vectorizer.fit_transform(data[column])
    return tfidf_matrix

tfidf_matrix = build_model(movies, text_col)

# =========================
# RECOMMENDER (USER PROFILE)
# =========================
def recommend_from_history(watched_titles, top_n=10):
    watched_idx = movies[movies["title"].isin(watched_titles)].index

    user_vector = np.mean(
        tfidf_matrix[watched_idx].toarray(),
        axis=0
    ).reshape(1, -1)

    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()
    similarity[watched_idx] = 0

    top_indices = similarity.argsort()[::-1][:top_n]
    results = movies.iloc[top_indices].copy()
    results["similarity"] = similarity[top_indices]

    return results

# =========================
# HEADER
# =========================
st.markdown("""
<div class="card" style="text-align:center;">
    <h1>🎬 MovieVerse</h1>
    <p style="color:#475569; font-size:15px;">
        Modern User-Based Movie Recommendation System
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🎥 Histori Tontonan")
    watched_movies = st.multiselect(
        "Pilih minimal 5 film yang pernah ditonton",
        movies["title"].values
    )

    min_watch = 5
    st.info("🎯 Sistem akan menampilkan **10 rekomendasi film**")

# =========================
# MAIN CONTENT
# =========================
if len(watched_movies) < min_watch:
    st.warning(f"⚠️ Pilih minimal **{min_watch} film** untuk mendapatkan rekomendasi")
else:
    st.markdown("""
    <div class="card">
        <span class="badge">PROFIL USER TERBENTUK</span>
        <p style="margin-top:12px;">
            Preferensi dihitung berdasarkan histori tontonan Anda
        </p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = recommend_from_history(watched_movies, top_n=10)

    st.subheader("✨ 10 Rekomendasi Film Untuk Anda")

    cols = st.columns(5)
    for i, row in recommendations.iterrows():
        with cols[i % 5]:
            st.markdown(f"""
            <div class="rec-card">
                <h4 style="font-size:14px; min-height:48px;">
                    {row['title']}
                </h4>
                <p style="font-size:12px; color:#64748b;">
                    Similarity: {row['similarity']:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
MovieVerse • User Profile Based Recommender  
<br>TF-IDF & Cosine Similarity
</div>
""", unsafe_allow_html=True)
