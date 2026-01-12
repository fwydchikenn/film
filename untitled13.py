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
# CSS (CINEMATIC)
# =========================
st.markdown("""
<style>
body { background-color: #0b0f19; }
.main { background-color: #0b0f19; }
h1, h2, h3 { color: #f8fafc; }
.card {
    background: #020617;
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 12px 30px rgba(0,0,0,0.5);
}
.rec-card {
    background: #020617;
    padding: 16px;
    border-radius: 14px;
    text-align: center;
}
.badge {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 12px;
    color: white;
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
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_model(movies, text_col)

# =========================
# RECOMMENDER (USER PROFILE)
# =========================
def recommend_from_history(watched_titles, top_n=6):
    watched_idx = movies[movies["title"].isin(watched_titles)].index

    # USER PROFILE VECTOR = RATA-RATA FILM YANG DITONTON
    user_vector = np.mean(tfidf_matrix[watched_idx].toarray(), axis=0).reshape(1, -1)

    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # HILANGKAN FILM YANG SUDAH DITONTON
    similarity[watched_idx] = 0

    top_indices = similarity.argsort()[::-1][:top_n]
    return movies.iloc[top_indices]

# =========================
# UI HEADER
# =========================
st.markdown("<h1>🎬 MovieVerse</h1>", unsafe_allow_html=True)
st.caption("User Watch History Based Recommendation")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🎥 Film yang Pernah Ditonton")
    watched_movies = st.multiselect(
        "Pilih minimal 5–10 film",
        movies["title"].values
    )

    min_watch = 5
    top_n = st.slider("Jumlah Rekomendasi", 3, 12, 6)

# =========================
# MAIN LOGIC
# =========================
if len(watched_movies) < min_watch:
    st.warning(f"⚠️ Pilih minimal **{min_watch} film** untuk mendapatkan rekomendasi")
else:
    st.markdown("""
    <div class="card">
        <span class="badge">PROFIL USER TERBENTUK</span>
        <p>Preferensi dihitung dari histori tontonan Anda</p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = recommend_from_history(watched_movies, top_n)

    st.subheader("✨ Rekomendasi Film Untuk Anda")

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
<div style="text-align:center;color:#64748b;margin-top:60px;">
MovieVerse • User Profile Based Recommender<br>
TF-IDF + Cosine Similarity
</div>
""", unsafe_allow_html=True)
