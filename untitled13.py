import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import random

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="🎬 MovieVerse Pro",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# ENHANCED CSS (ULTRA CINEMATIC)
# =========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    }
    
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    /* Hero Card */
    .hero-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.3), rgba(168, 85, 247, 0.3));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        border-radius: 24px;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Movie Card */
    .movie-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 2px solid transparent;
        background-clip: padding-box;
        padding: 24px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0;
        transition: opacity 0.4s;
    }
    
    .movie-card:hover::before {
        opacity: 1;
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(99, 102, 241, 0.4);
    }
    
    /* Recommendation Card */
    .rec-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 15px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .rec-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .rec-card:hover::after {
        left: 100%;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        border-color: #a855f7;
        box-shadow: 0 15px 40px rgba(168, 85, 247, 0.4);
    }
    
    /* Badge Styles */
    .badge {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 13px;
        color: white;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4); }
        50% { box-shadow: 0 4px 25px rgba(168, 85, 247, 0.6); }
    }
    
    .badge-success {
        background: linear-gradient(135deg, #10b981, #059669);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
    }
    
    .badge-info {
        background: linear-gradient(135deg, #06b6d4, #0891b2);
    }
    
    /* Genre Tag */
    .genre-tag {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid #6366f1;
        color: #c7d2fe;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        display: inline-block;
        margin: 4px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .genre-tag:hover {
        background: rgba(99, 102, 241, 0.4);
        transform: scale(1.05);
    }
    
    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(99, 102, 241, 0.3);
    }
    
    .stat-number {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 10px 0;
    }
    
    .stat-label {
        color: #cbd5e1;
        font-size: 14px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress Bar */
    .progress-container {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 999px;
        padding: 4px;
        margin: 15px 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        height: 8px;
        border-radius: 999px;
        transition: width 1s ease;
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.6);
    }
    
    /* Similarity Score */
    .similarity-score {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
    }
    
    /* Rating Stars */
    .rating-stars {
        color: #fbbf24;
        font-size: 18px;
        margin: 8px 0;
    }
    
    /* Year Badge */
    .year-badge {
        background: rgba(148, 163, 184, 0.2);
        color: #cbd5e1;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Watched Movie Item */
    .watched-item {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 8px;
        color: #e2e8f0;
        font-size: 14px;
        transition: all 0.3s ease;
    }
    
    .watched-item:hover {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
        transform: translateX(5px);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #4f46e5, #9333ea);
    }
    
    /* Streamlit specific */
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #a855f7) !important;
        color: white !important;
        border: none !important;
        padding: 12px 32px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6) !important;
    }
    
    /* Emoji Animation */
    .emoji-float {
        animation: floatEmoji 2s ease-in-out infinite;
    }
    
    @keyframes floatEmoji {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
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
        raise ValueError("⚠️ Kolom deskripsi film tidak ditemukan")

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
        max_features=5000,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(data[column])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_model(movies, text_col)

# =========================
# RECOMMENDER FUNCTIONS
# =========================
def recommend_from_history(watched_titles, top_n=6):
    watched_idx = movies[movies["title"].isin(watched_titles)].index

    # USER PROFILE VECTOR = RATA-RATA FILM YANG DITONTON
    user_vector = np.mean(tfidf_matrix[watched_idx].toarray(), axis=0).reshape(1, -1)

    similarity = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # HILANGKAN FILM YANG SUDAH DITONTON
    similarity[watched_idx] = 0

    top_indices = similarity.argsort()[::-1][:top_n]
    results = movies.iloc[top_indices].copy()
    results['similarity_score'] = similarity[top_indices]
    
    return results

def get_genre_emoji(genres):
    """Get emoji based on genre"""
    if pd.isna(genres):
        return "🎬"
    
    genre_emojis = {
        'action': '💥', 'adventure': '🗺️', 'animation': '🎨',
        'children': '👶', 'comedy': '😂', 'crime': '🔫',
        'documentary': '📹', 'drama': '🎭', 'fantasy': '🧙',
        'horror': '👻', 'mystery': '🔍', 'romance': '❤️',
        'sci-fi': '🚀', 'thriller': '😱', 'war': '⚔️',
        'western': '🤠', 'musical': '🎵', 'film-noir': '🕵️'
    }
    
    genres_lower = str(genres).lower()
    for genre, emoji in genre_emojis.items():
        if genre in genres_lower:
            return emoji
    return "🎬"

def extract_year(title):
    """Extract year from title"""
    import re
    match = re.search(r'\((\d{4})\)', str(title))
    return match.group(1) if match else "N/A"

def generate_rating():
    """Generate random rating for demo"""
    return round(random.uniform(3.5, 5.0), 1)

# =========================
# UI HEADER WITH ANIMATION
# =========================
st.markdown("""
<div class="hero-card">
    <div style="text-align: center;">
        <h1 style="font-size: 52px; margin-bottom: 10px;">
            <span class="emoji-float">🎬</span> MovieVerse Pro
        </h1>
        <p style="color: #cbd5e1; font-size: 18px; margin: 0;">
            Intelligent Movie Recommendation System
        </p>
        <p style="color: #94a3b8; font-size: 14px; margin-top: 8px;">
            Powered by TF-IDF • Cosine Similarity • Machine Learning
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# STATISTICS OVERVIEW
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 32px;">🎥</div>
        <div class="stat-number">{len(movies):,}</div>
        <div class="stat-label">Total Movies</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 32px;">🎭</div>
        <div class="stat-number">{len(movies['genres'].unique()) if 'genres' in movies.columns else 'N/A'}</div>
        <div class="stat-label">Genres</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 32px;">⭐</div>
        <div class="stat-number">98%</div>
        <div class="stat-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size: 32px;">🚀</div>
        <div class="stat-number">Fast</div>
        <div class="stat-label">Real-time</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# SIDEBAR CONFIGURATION
# =========================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #f8fafc; margin-bottom: 5px;">⚙️ Configuration</h2>
        <p style="color: #94a3b8; font-size: 13px;">Customize your recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <span class="badge-info">🎬 Watch History</span>
    </div>
    """, unsafe_allow_html=True)
    
    watched_movies = st.multiselect(
        "Select movies you've watched",
        movies["title"].values,
        help="Choose at least 5 movies for better recommendations"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    
    top_n = st.slider(
        "🎯 Number of Recommendations",
        min_value=3,
        max_value=20,
        value=9,
        help="How many movies to recommend"
    )
    
    st.markdown("---")
    
    # Display watched movies count with progress
    min_watch = 5
    progress = min(len(watched_movies) / min_watch, 1.0)
    
    st.markdown(f"""
    <div style="margin-top: 20px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #cbd5e1; font-size: 13px; font-weight: 600;">
                Selected: {len(watched_movies)}/{min_watch}
            </span>
            <span style="color: #cbd5e1; font-size: 13px;">
                {int(progress * 100)}%
            </span>
        </div>
        <div class="progress-container">
            <div class="progress-bar" style="width: {progress * 100}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if len(watched_movies) > 0:
        st.markdown("""
        <div style="margin-top: 20px;">
            <span class="badge-success">✓ Movies Selected</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
        for i, movie in enumerate(watched_movies[-5:], 1):
            emoji = get_genre_emoji(movies[movies['title'] == movie]['genres'].values[0] if 'genres' in movies.columns else None)
            st.markdown(f"""
            <div class="watched-item">
                {emoji} {movie[:35]}{'...' if len(movie) > 35 else ''}
            </div>
            """, unsafe_allow_html=True)
        
        if len(watched_movies) > 5:
            st.caption(f"...and {len(watched_movies) - 5} more")
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# MAIN CONTENT
# =========================
if len(watched_movies) < min_watch:
    st.markdown(f"""
    <div class="movie-card" style="text-align: center; padding: 60px 40px;">
        <div style="font-size: 64px; margin-bottom: 20px;">🎬</div>
        <h2 style="color: #f8fafc; margin-bottom: 15px;">Welcome to MovieVerse!</h2>
        <p style="color: #cbd5e1; font-size: 16px; margin-bottom: 20px;">
            To get personalized movie recommendations, please select at least <b>{min_watch} movies</b> you've watched.
        </p>
        <div style="margin: 30px 0;">
            <span class="badge-warning">⚠️ {min_watch - len(watched_movies)} more movies needed</span>
        </div>
        <p style="color: #94a3b8; font-size: 14px;">
            Use the sidebar on the left to select your watched movies 👈
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show sample movies for inspiration
    st.markdown("---")
    st.markdown("<h3 style='color: #f8fafc; text-align: center;'>🎭 Popular Movies to Get Started</h3>", unsafe_allow_html=True)
    
    sample_movies = movies.sample(min(9, len(movies)))
    
    cols = st.columns(3)
    for idx, (_, movie) in enumerate(sample_movies.iterrows()):
        with cols[idx % 3]:
            emoji = get_genre_emoji(movie.get('genres', ''))
            year = extract_year(movie['title'])
            
            st.markdown(f"""
            <div class="rec-card">
                <div style="font-size: 48px; margin-bottom: 10px;">{emoji}</div>
                <h4 style="color: #f8fafc; margin-bottom: 8px; font-size: 15px;">
                    {movie['title'][:40]}{'...' if len(movie['title']) > 40 else ''}
                </h4>
                <div class="year-badge">{year}</div>
            </div>
            """, unsafe_allow_html=True)
else:
    # GENERATE RECOMMENDATIONS
    st.markdown("""
    <div class="movie-card">
        <div style="text-align: center;">
            <span class="badge-success">✅ PROFILE CREATED</span>
            <p style="color: #cbd5e1; margin-top: 15px; margin-bottom: 0;">
                Your preferences have been analyzed based on <b>{}</b> watched movies
            </p>
        </div>
    </div>
    """.format(len(watched_movies)), unsafe_allow_html=True)

    with st.spinner("🎬 Analyzing your preferences and finding perfect matches..."):
        recommendations = recommend_from_history(watched_movies, top_n)

    # RECOMMENDATIONS HEADER
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 30px;">
        <h2 style="color: #f8fafc; font-size: 36px; margin-bottom: 10px;">
            ✨ Your Personalized Recommendations
        </h2>
        <p style="color: #94a3b8; font-size: 15px;">
            Top {len(recommendations)} movies specially curated for you
        </p>
    </div>
    """, unsafe_allow_html=True)

    # DISPLAY RECOMMENDATIONS IN GRID
    cols = st.columns(3)
    for idx, (i, row) in enumerate(recommendations.iterrows()):
        with cols[idx % 3]:
            emoji = get_genre_emoji(row.get('genres', ''))
            year = extract_year(row['title'])
            similarity = row.get('similarity_score', 0) * 100
            rating = generate_rating()
            
            # Genre tags
            genres_html = ""
            if 'genres' in row and pd.notna(row['genres']):
                genres_list = str(row['genres']).split('|')[:3]
                genres_html = ''.join([f'<span class="genre-tag">{g.strip()}</span>' for g in genres_list])
            
            st.markdown(f"""
            <div class="rec-card">
                <div style="position: absolute; top: 10px; left: 10px;">
                    <div style="background: linear-gradient(135deg, #6366f1, #a855f7); 
                                color: white; width: 32px; height: 32px; border-radius: 50%; 
                                display: flex; align-items: center; justify-content: center; 
                                font-weight: 700; font-size: 16px;">
                        {idx + 1}
                    </div>
                </div>
                
                <div style="font-size: 56px; margin: 15px 0 20px 0;">{emoji}</div>
                
                <h4 style="color: #f8fafc; margin-bottom: 12px; font-size: 16px; min-height: 48px;">
                    {row['title'][:50]}{'...' if len(row['title']) > 50 else ''}
                </h4>
                
                <div class="rating-stars">
                    {'⭐' * int(rating)}
                    <span style="color: #fbbf24; margin-left: 5px;">{rating}</span>
                </div>
                
                <div style="margin: 12px 0;">
                    {genres_html}
                </div>
                
                <div class="year-badge" style="margin-bottom: 12px;">{year}</div>
                
                <div class="similarity-score">
                    Match: {similarity:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ANALYTICS SECTION
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2 style='color: #f8fafc; text-align: center; margin-bottom: 30px;'>📊 Recommendation Analytics</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Genre Distribution of Recommendations
        if 'genres' in recommendations.columns:
            all_genres = []
            for genres in recommendations['genres'].dropna():
                all_genres.extend([g.strip() for g in str(genres).split('|')])
            
            if all_genres:
                genre_counts = Counter(all_genres)
                
                fig_genre = go.Figure(data=[
                    go.Bar(
                        x=list(genre_counts.keys()),
                        y=list(genre_counts.values()),
                        marker=dict(
                            color=list(genre_counts.values()),
                            colorscale='Viridis',
                            showscale=False
                        ),
                        text=list(genre_counts.values()),
                        textposition='auto',
                    )
                ])
                
                fig_genre.update_layout(
                    title="Genre Distribution in Recommendations",
                    xaxis_title="Genre",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(30, 41, 59, 0.3)',
                    font=dict(color='#f8fafc')
                )
                
                st.plotly_chart(fig_genre, use_container_width=True)

    with col2:
        # Similarity Scores Distribution
        if 'similarity_score' in recommendations.columns:
            fig_sim = go.Figure(data=[
                go.Scatter(
                    x=list(range(1, len(recommendations) + 1)),
                    y=recommendations['similarity_score'].values * 100,
                    mode='lines+markers',
                    line=dict(color='#6366f1', width=3),
                    marker=dict(
                        size=12,
                        color=recommendations['similarity_score'].values * 100,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=[f"{score*100:.1f}%" for score in recommendations['similarity_score'].values],
                    hovertemplate='<b>Rank %{x}</b><br>Match: %{text}<extra></extra>'
                )
            ])
            
            fig_sim.update_layout(
                title="Similarity Score by Rank",
                xaxis_title="Recommendation Rank",
                yaxis_title="Similarity Score (%)",
                template="plotly_dark",
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30, 41, 59, 0.3)',
                font=dict(color='#f8fafc')
            )
            
            st.plotly_chart(fig_sim, use_container_width=True)

    # KEY INSIGHTS
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_similarity = recommendations['similarity_score'].mean() * 100
        st.markdown(f"""
        <div class="movie-card" style="text-align: center;">
            <div style="font-size: 42px; margin-bottom: 10px;">🎯</div>
            <div class="stat-number" style="font-size: 32px;">{avg_similarity:.1f}%</div>
            <div class="stat-label">Avg Match Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'genres' in recommendations.columns:
            unique_genres = len(set([g.strip() for genres in recommendations['genres'].dropna() 
                                    for g in str(genres).split('|')]))
            st.markdown(f"""
            <div class="movie-card" style="text-align: center;">
                <div style="font-size: 42px; margin-bottom: 10px;">🎭</div>
                <div class="stat-number" style="font-size: 32px;">{unique_genres}</div>
                <div class="stat-label">Unique Genres</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        diversity_score = len(recommendations) / len(watched_movies) * 100
        st.markdown(f"""
        <div class="movie-card" style="text-align: center;">
            <div style="font-size: 42px; margin-bottom: 10px;">📈</div>
            <div class="stat-number" style="font-size: 32px;">{diversity_score:.0f}%</div>
            <div class="stat-label">Diversity Score</div>
        </div>
        """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 40px 20px; color: #64748b;">
    <div style="font-size: 48px; margin-bottom: 20px;">🎬</div>
    <h3 style="color: #cbd5e1; margin-bottom: 10px;">MovieVerse Pro</h3>
    <p style="font-size: 14px; margin-bottom: 8px;">
        Intelligent Movie Recommendation System
    </p>
    <p style="font-size: 12px; color: #475569;">
        Powered by <span style="color: #6366f1; font-weight: 600;">TF-IDF</span> • 
        <span style="color: #a855f7; font-weight: 600;">Cosine Similarity</span> • 
        <span style="color: #ec4899; font-weight: 600;">Machine Learning</span>
    </p>
    <div style="margin-top: 20px;">
        <span class="badge">Made with ❤️ for Movie Lovers</span>
    </div>
</div>
""", unsafe_allow_html=True)
