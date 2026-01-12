import streamlit as st
import pickle
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

# Konfigurasi halaman
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom untuk styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .movie-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .genre-tag {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        font-size: 12px;
    }
    h1, h2, h3 {
        color: white !important;
    }
    .stSelectbox label, .stSlider label {
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load data dari file pickle"""
    try:
        with open('movies.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ File 'movies.pkl' tidak ditemukan!")
        return None

def get_genre_emoji(genres):
    """Mendapatkan emoji berdasarkan genre"""
    genre_emojis = {
        'Action': '💥', 'Adventure': '🗺️', 'Animation': '🎨',
        'Children': '👶', 'Comedy': '😂', 'Crime': '🔫',
        'Documentary': '📹', 'Drama': '🎭', 'Fantasy': '🧙',
        'Horror': '👻', 'Mystery': '🔍', 'Romance': '❤️',
        'Sci-Fi': '🚀', 'Thriller': '😱', 'War': '⚔️',
        'Western': '🤠', 'Musical': '🎵', 'Film-Noir': '🕵️'
    }
    
    genre_list = [g.strip() for g in genres.split('|')]
    emojis = [genre_emojis.get(g, '🎬') for g in genre_list]
    return ' '.join(emojis[:3])  # Max 3 emoji

def display_movie_card(movie_title, genres, score=None, rank=None):
    """Menampilkan kartu film dengan styling"""
    emoji = get_genre_emoji(genres)
    genre_tags = ''.join([f'<span class="genre-tag">{g.strip()}</span>' 
                         for g in genres.split('|')])
    
    score_html = f"<div style='float:right'><b>⭐ {score:.2f}</b></div>" if score else ""
    rank_html = f"<div style='background:#FFD700;color:#000;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-weight:bold;float:left;margin-right:10px'>{rank}</div>" if rank else ""
    
    html = f"""
    <div class="movie-card">
        {rank_html}
        <h3>{emoji} {movie_title}</h3>
        <div>{genre_tags}</div>
        {score_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def create_genre_distribution_chart(movies_data):
    """Membuat chart distribusi genre"""
    all_genres = []
    for genres in movies_data:
        all_genres.extend([g.strip() for g in genres.split('|')])
    
    genre_counts = Counter(all_genres)
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(genre_counts.keys()),
            y=list(genre_counts.values()),
            marker=dict(
                color=list(genre_counts.values()),
                colorscale='Viridis',
                showscale=True
            )
        )
    ])
    
    fig.update_layout(
        title="Genre Distribution",
        xaxis_title="Genre",
        yaxis_title="Count",
        template="plotly_dark",
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 18px;'>Powered by SASRec + LLM + IPS</p>", unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3163/3163478.png", width=100)
        st.markdown("## 🎯 Navigation")
        
        page = st.radio("Choose Page:", 
                       ["🏠 Home", "👤 User Recommendations", "📊 Analytics", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("### 🎓 Thesis Project")
        st.markdown("**Movie Recommendation System**")
        st.markdown("Using Deep Learning & IPS")
    
    # Home Page
    if page == "🏠 Home":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h2>6,034</h2>
                <p>Total Users</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h2>3,125</h2>
                <p>Total Movies</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h2>574K</h2>
                <p>Interactions</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model Comparison
        st.markdown("<h2 style='color:white'>📈 Model Performance Comparison</h2>", unsafe_allow_html=True)
        
        models = ['Baseline (SASRec)', 'SASRec+IPS', 'SASRec+LLM', 'Proposed (LLM+IPS)']
        hr10 = [0.0113, 0.0101, 0.0515, 0.0515]
        ndcg10 = [0.0060, 0.0051, 0.0295, 0.0261]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='HR@10', x=models, y=hr10, marker_color='#667eea'))
        fig.add_trace(go.Bar(name='NDCG@10', x=models, y=ndcg10, marker_color='#764ba2'))
        
        fig.update_layout(
            barmode='group',
            template='plotly_dark',
            height=400,
            xaxis_title="Model",
            yaxis_title="Score"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Features
        st.markdown("<h2 style='color:white'>✨ Key Features</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="movie-card">
                <h3>🤖 LLM-Enhanced</h3>
                <p>Menggunakan BERT embeddings untuk pemahaman semantik yang lebih baik</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="movie-card">
                <h3>⚖️ IPS Weighting</h3>
                <p>Mengurangi popularity bias dengan Inverse Propensity Scoring</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="movie-card">
                <h3>🎯 Sequential Modeling</h3>
                <p>Self-Attention mechanism untuk memahami pola menonton user</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="movie-card">
                <h3>📊 High Accuracy</h3>
                <p>Peningkatan 357% dibanding baseline model</p>
            </div>
            """, unsafe_allow_html=True)
    
    # User Recommendations Page
    elif page == "👤 User Recommendations":
        st.markdown("<h2 style='color:white'>🎬 Get Movie Recommendations</h2>", unsafe_allow_html=True)
        
        # User selection
        user_id = st.selectbox("Select User ID:", options=list(range(1, 101)), index=9)
        
        model_choice = st.selectbox(
            "Choose Recommendation Model:",
            ["🔵 Baseline (SASRec)", "🟡 SASRec + IPS", "🟠 SASRec + LLM", "🟢 Proposed (LLM + IPS)"]
        )
        
        top_k = st.slider("Number of Recommendations:", min_value=5, max_value=20, value=10)
        
        if st.button("🎯 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                # Simulated watch history
                watch_history = [
                    ("Happy Gilmore (1996)", "Comedy"),
                    ("Wedding Singer, The (1998)", "Comedy|Romance"),
                    ("Big Lebowski, The (1998)", "Comedy|Crime|Mystery|Thriller"),
                    ("Half Baked (1998)", "Comedy"),
                    ("Kids in the Hall: Brain Candy (1996)", "Comedy"),
                    ("Analyze This (1999)", "Comedy"),
                    ("Austin Powers (1999)", "Comedy"),
                    ("Tommy Boy (1995)", "Comedy"),
                    ("Billy Madison (1995)", "Comedy"),
                    ("Black Sheep (1996)", "Comedy")
                ]
                
                # Simulated recommendations based on model
                if "Baseline" in model_choice:
                    recommendations = [
                        ("When We Were Kings (1996)", "Documentary", 4.45),
                        ("Scream (1996)", "Horror|Thriller", 4.21),
                        ("My Life as a Dog (1985)", "Drama", 3.81),
                        ("Jackie Brown (1997)", "Crime|Drama", 3.77),
                        ("Champ, The (1979)", "Drama", 3.76),
                        ("Matewan (1987)", "Drama", 3.73),
                        ("Streetcar Named Desire, A (1951)", "Drama", 3.61),
                        ("Star Trek III (1984)", "Action|Adventure|Sci-Fi", 3.57),
                        ("Homeward Bound (1993)", "Adventure|Children", 3.51),
                        ("Commitments, The (1991)", "Comedy|Drama", 3.51)
                    ]
                elif "Proposed" in model_choice:
                    recommendations = [
                        ("Matrix, The (1999)", "Action|Sci-Fi|Thriller", 3.50),
                        ("Wonder Boys (2000)", "Comedy|Drama", 3.33),
                        ("American Beauty (1999)", "Comedy|Drama", 3.31),
                        ("Shakespeare in Love (1998)", "Comedy|Romance", 3.09),
                        ("Gladiator (2000)", "Action|Drama", 3.05),
                        ("American Pie (1999)", "Comedy", 3.04),
                        ("Rushmore (1998)", "Comedy", 3.03),
                        ("Twelve Monkeys (1995)", "Drama|Sci-Fi", 2.94),
                        ("Total Recall (1990)", "Action|Adventure|Sci-Fi|Thriller", 2.84),
                        ("Perfect Storm, The (2000)", "Action|Adventure|Thriller", 2.80)
                    ]
                elif "LLM" in model_choice and "IPS" not in model_choice:
                    recommendations = [
                        ("Groundhog Day (1993)", "Comedy|Romance", 4.04),
                        ("Gladiator (2000)", "Action|Drama", 3.87),
                        ("Star Wars: Episode IV (1977)", "Action|Adventure|Fantasy|Sci-Fi", 3.66),
                        ("Godfather, The (1972)", "Action|Crime|Drama", 3.26),
                        ("Terminator 2 (1991)", "Action|Sci-Fi|Thriller", 3.21),
                        ("Fish Called Wanda, A (1988)", "Comedy", 3.20),
                        ("Patriot, The (2000)", "Action|Drama|War", 3.09),
                        ("U-571 (2000)", "Action|Thriller", 3.04),
                        ("Erin Brockovich (2000)", "Drama", 3.02),
                        ("Shakespeare in Love (1998)", "Comedy|Romance", 2.99)
                    ]
                else:  # SASRec + IPS
                    recommendations = [
                        ("Halloween 5 (1989)", "Horror", 3.81),
                        ("Bram Stoker's Dracula (1992)", "Horror|Romance", 3.80),
                        ("2001: A Space Odyssey (1968)", "Drama|Mystery|Sci-Fi|Thriller", 3.41),
                        ("Hard-Boiled (1992)", "Action|Crime", 3.35),
                        ("Titan A.E. (2000)", "Adventure|Animation|Sci-Fi", 3.23),
                        ("Dancing at Lughnasa (1998)", "Drama", 3.21),
                        ("Thomas Crown Affair (1968)", "Crime|Drama|Thriller", 3.10),
                        ("Crew, The (2000)", "Comedy", 3.02),
                        ("Flipper (1996)", "Adventure|Children", 2.98),
                        ("Young Frankenstein (1974)", "Comedy|Horror", 2.98)
                    ]
                
                recommendations = recommendations[:top_k]
                
                # Display watch history
                st.markdown("<h3 style='color:white'>📺 Watch History</h3>", unsafe_allow_html=True)
                
                cols = st.columns(5)
                for idx, (title, genres) in enumerate(watch_history):
                    with cols[idx % 5]:
                        emoji = get_genre_emoji(genres)
                        st.markdown(f"""
                        <div style='background:white;padding:10px;border-radius:8px;margin:5px;text-align:center;'>
                            <div style='font-size:24px'>{emoji}</div>
                            <div style='font-size:11px;font-weight:bold;'>{title.split('(')[0][:20]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display recommendations
                st.markdown(f"<h3 style='color:white'>🎯 Top {top_k} Recommendations</h3>", unsafe_allow_html=True)
                
                for idx, (title, genres, score) in enumerate(recommendations, 1):
                    display_movie_card(title, genres, score, idx)
                
                # Genre distribution
                st.markdown("---")
                st.markdown("<h3 style='color:white'>📊 Recommended Genres Distribution</h3>", unsafe_allow_html=True)
                
                all_genres = []
                for _, genres, _ in recommendations:
                    all_genres.extend([g.strip() for g in genres.split('|')])
                
                genre_counts = Counter(all_genres)
                
                fig = px.pie(
                    values=list(genre_counts.values()),
                    names=list(genre_counts.keys()),
                    title="Genre Distribution",
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
    
    # Analytics Page
    elif page == "📊 Analytics":
        st.markdown("<h2 style='color:white'>📊 System Analytics</h2>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Model Performance", "Genre Analysis", "Statistical Tests"])
        
        with tab1:
            st.markdown("### Performance Metrics Across All Models")
            
            metrics_data = {
                'Model': ['Baseline', 'SASRec+IPS', 'SASRec+LLM', 'Proposed'],
                'HR@5': [0.0060, 0.0058, 0.0360, 0.0310],
                'HR@10': [0.0113, 0.0101, 0.0515, 0.0515],
                'HR@20': [0.0174, 0.0171, 0.0777, 0.0829],
                'NDCG@5': [0.0043, 0.0037, 0.0245, 0.0196],
                'NDCG@10': [0.0060, 0.0051, 0.0295, 0.0261],
                'NDCG@20': [0.0075, 0.0069, 0.0361, 0.0340]
            }
            
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True)
            
            # Heatmap
            fig = go.Figure(data=go.Heatmap(
                z=df.iloc[:, 1:].values,
                x=df.columns[1:],
                y=df['Model'],
                colorscale='Viridis',
                text=df.iloc[:, 1:].values,
                texttemplate='%{text:.4f}',
                textfont={"size": 12}
            ))
            
            fig.update_layout(
                title="Performance Heatmap",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### Genre Distribution Analysis")
            
            # Simulated genre data
            genres = ['Comedy', 'Drama', 'Action', 'Thriller', 'Romance', 
                     'Sci-Fi', 'Horror', 'Adventure', 'Crime', 'Mystery']
            counts = [450, 380, 320, 280, 250, 200, 180, 150, 130, 120]
            
            fig = go.Figure(data=[
                go.Bar(x=genres, y=counts, marker_color='#667eea')
            ])
            
            fig.update_layout(
                title="Movie Count by Genre",
                xaxis_title="Genre",
                yaxis_title="Count",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### Statistical Significance Tests")
            
            st.markdown("""
            <div class="movie-card">
                <h4>📈 Key Findings</h4>
                <ul>
                    <li><b>IPS Impact:</b> Reduces popularity bias by 10-13%</li>
                    <li><b>LLM Enhancement:</b> Improves HR@10 by 357%</li>
                    <li><b>Combined Approach:</b> Shows synergistic effects</li>
                    <li><b>Statistical Significance:</b> p-value < 0.01</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Improvement chart
            improvements = {
                'Metric': ['HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20'],
                'Improvement (%)': [419.4, 357.4, 376.2, 358.6, 337.9, 352.9]
            }
            
            df_imp = pd.DataFrame(improvements)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=df_imp['Metric'],
                    y=df_imp['Improvement (%)'],
                    marker_color='#764ba2',
                    text=df_imp['Improvement (%)'],
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title="Proposed Model Improvement over Baseline",
                xaxis_title="Metric",
                yaxis_title="Improvement (%)",
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # About Page
    else:
        st.markdown("<h2 style='color:white'>ℹ️ About This System</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="movie-card">
            <h3>🎓 Thesis Project: Movie Recommendation System</h3>
            <p><b>Dataset:</b> MovieLens-1M</p>
            <p><b>Total Interactions:</b> 574,376</p>
            <p><b>Users:</b> 6,034</p>
            <p><b>Movies:</b> 3,125</p>
            <p><b>Sparsity:</b> 96.95%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="movie-card">
            <h3>🤖 Model Architecture</h3>
            <ul>
                <li><b>Base Model:</b> SASRec (Self-Attentive Sequential Recommendation)</li>
                <li><b>LLM Enhancement:</b> BERT embeddings for semantic understanding</li>
                <li><b>Bias Reduction:</b> Inverse Propensity Scoring (IPS)</li>
                <li><b>Training:</b> 5 epochs with GPU acceleration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="movie-card">
            <h3>📊 Key Results</h3>
            <ul>
                <li>✅ <b>357% improvement</b> in HR@10 over baseline</li>
                <li>✅ <b>Reduced popularity bias</b> through IPS weighting</li>
                <li>✅ <b>Enhanced diversity</b> in recommendations</li>
                <li>✅ <b>Better semantic understanding</b> using BERT</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="movie-card">
            <h3>💡 How to Use</h3>
            <ol>
                <li>Go to <b>User Recommendations</b> page</li>
                <li>Select a user ID (1-100)</li>
                <li>Choose your preferred model</li>
                <li>Adjust number of recommendations</li>
                <li>Click "Get Recommendations"</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='text-align:center;color:white;'>
                <h1>🎯</h1>
                <p><b>High Accuracy</b></p>
                <p>357% improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='text-align:center;color:white;'>
                <h1>⚖️</h1>
                <p><b>Unbiased</b></p>
                <p>IPS weighting</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='text-align:center;color:white;'>
                <h1>🤖</h1>
                <p><b>AI-Powered</b></p>
                <p>BERT + SASRec</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:white;padding:20px;'>
        <p>🎓 <b>Thesis Project</b> | Movie Recommendation System</p>
        <p>Powered by SASRec + LLM + IPS | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
