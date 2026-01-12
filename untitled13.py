import streamlit as st
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG (LIGHT)
# =========================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# =========================
# LIGHT UI CSS
# =========================
st.markdown("""
<style>
body { background-color: #f8fafc; }
.main { background-color: #f8fafc; }
h1,h2,h3 { color:#0f172a; }
.card {
    background:white;
    padding:20px;
    border-radius:14px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom:20px;
}
.rec-card {
    background:white;
    padding:16px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 6px 18px rgba(0,0,0,0.08);
}
.badge {
    background:#2563eb;
    color:white;
    padding:5px 12px;
    border-radius:999px;
    font-size:12px;
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
        raise ValueError("Kolom deskripsi tidak ditemukan")

    df[text_col] = df[text_col].fillna("")
    df["genres"] = df.get("genres", "").fillna("")
    df["release_date"] = pd.to_datetime(
        df.get("release_date", "2000-01-01"), errors="coerce"
    )

    df["combined_text"] = df[text_col] + " " + df["genres"]
    return df.reset_index(drop=True)

movies = load_movies()

# =========================
# TF-IDF
# =========================
@st.cache_resource
def build_tfidf(data):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=6000
    )
    matrix = tfidf.fit_transform(data["combined_text"])
    return matrix

tfidf_matrix = build_tfidf(movies)

# =========================
# TIME DECAY
# =========================
def time_decay(release_date, alpha=0.03):
    if pd.isna(release_date):
        return 1
    age = (datetime.now() - release_date).days / 365
    return 1 / (1 + alpha * age)

# =========================
# USER PROFILE RECOMMENDER
# =========================
def recommend_user_profile(history, ratings, top_k=6):
    vectors = []
    weights = []

    for title, rating in zip(history, ratings):
        idx = movies[movies["title"] == title].index[0]
        decay = time_decay(movies.loc[idx, "release_date"])
        weight = rating * decay
        vectors.append(tfidf_matrix[idx].toarray()[0] * weight)
        weights.append(weight)

    user_vector = np.sum(vectors, axis=0) / np.sum(weights)
    sims = cosine_similarity(user_vector.reshape(1, -1), tfidf_matrix).flatten()

    watched_idx = movies[movies["title"].isin(history)].index
    sims[watched_idx] = 0

    top_idx = sims.argsort()[::-1][:top_k]
    return movies.iloc[top_idx]

# =========================
# EVALUATION (PRECISION@K)
# =========================
def precision_at_k(recommended, relevant):
    if len(recommended) == 0:
        return 0
    hit = len(set(recommended) & set(relevant))
    return hit / len(recommended)

# =========================
# UI HEADER
# =========================
st.title("🎬 Movie Recommendation System")
st.caption("Hybrid Content-Based | Weighted User Profile")

# =========================
# SIDEBAR INPUT
# =========================
with st.sidebar:
    st.header("🎥 Histori Tontonan")
    watched = st.multiselect(
        "Pilih minimal 5 film",
        movies["title"].values
    )

    ratings = []
    for w in watched:
        ratings.append(
            st.slider(f"Rating untuk {w}", 1, 5, 4)
        )

    top_k = st.slider("Jumlah Rekomendasi", 3, 12, 6)

# =========================
# MAIN LOGIC
# =========================
if len(watched) < 5:
    st.warning("⚠️ Pilih minimal 5 film untuk membangun profil user")
else:
    st.markdown("""
    <div class="card">
        <span class="badge">USER PROFILE TERBENTUK</span>
        <p>Rekomendasi berdasarkan histori, rating, genre, dan waktu rilis</p>
    </div>
    """, unsafe_allow_html=True)

    recs = recommend_user_profile(watched, ratings, top_k)

    st.subheader("✨ Rekomendasi Untuk Anda")

    cols = st.columns(3)
    for i, row in recs.iterrows():
        with cols[i % 3]:
            st.markdown(f"""
            <div class="rec-card">
                <h4>{row['title']}</h4>
                <p>{row['genres']}</p>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # EVALUATION
    # =========================
    st.subheader("📊 Evaluasi (Precision@K)")
    precision = precision_at_k(
        recs["title"].tolist(),
        watched
    )
    st.metric("Precision@K", round(precision, 2))

# =========================
# FOOTER
# =========================
st.markdown("""
<div style="text-align:center;color:#64748b;margin-top:60px;">
Hybrid Recommendation System<br>
TF-IDF • User Rating • Time Decay • Genre
</div>
""", unsafe_allow_html=True)
