import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import time
from PIL import Image
import requests
from io import BytesIO
import difflib

# Import your RecommenderSystem
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    from src.recommenders.recommender_system import RecommenderSystem
except ImportError as e:
    st.error(f"Error importing RecommenderSystem: {str(e)}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .movie-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .overview-text {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .suggestion-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .poster-container {
        text-align: center;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
    st.session_state.is_initialized = False
    st.session_state.selected_movies = []

def get_movie_poster(movie_row):
    """Get movie poster URL from the data or return placeholder"""
    try:
        # Check if poster_path exists and is not null
        if 'poster_path' in movie_row and pd.notna(movie_row['poster_path']) and movie_row['poster_path']:
            poster_path = str(movie_row['poster_path']).strip()
            
            # If poster_path starts with '/', it's a TMDb path
            if poster_path.startswith('/'):
                base_url = "https://image.tmdb.org/t/p/w300"
                poster_url = f"{base_url}{poster_path}"
                return poster_url
            elif poster_path.startswith('http'):
                return poster_path
            else:
                base_url = "https://image.tmdb.org/t/p/w300"
                poster_url = f"{base_url}/{poster_path}"
                return poster_url
        
        # Fallback placeholder
        title = movie_row.get('title', 'Movie')[:15]
        placeholder_url = f"https://via.placeholder.com/300x450/667eea/white?text={title.replace(' ', '+')}"
        return placeholder_url
        
    except Exception as e:
        return "https://via.placeholder.com/300x450/dc3545/white?text=Error+Loading+Poster"

def load_poster_image(poster_url, width=150):
    """Load and display poster image with error handling"""
    try:
        response = requests.get(poster_url, timeout=5)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            return image
        else:
            return None
    except:
        return None

def initialize_recommender(enable_svd=True):
    """Initialize the recommender system with progress tracking"""
    try:
        # Get the absolute path of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define relative paths to the data files
        ratings_path = os.path.join(current_dir, 'data', 'processed', 'ratings_cleans.csv')
        metadata_path = os.path.join(current_dir, 'data', 'processed', 'movies_with_sentiment.csv')        
        # Check if files exist
        if not os.path.exists(ratings_path):
            st.error(f"❌ Ratings file not found: {ratings_path}")
            return False
        if not os.path.exists(metadata_path):
            st.error(f"❌ Metadata file not found: {metadata_path}")
            return False
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🔄 Initializing RecommenderSystem...")
        progress_bar.progress(10)
        
        recommender = RecommenderSystem(
            ratings_path=ratings_path,
            metadata_path=metadata_path,
            content_cols=['overview', 'tagline', 'genres', 'cast', 'crew', 'keywords'],
            verbose=False,
            enable_svd=enable_svd
        )
        
        status_text.text("📊 Loading data...")
        progress_bar.progress(20)
        recommender.load_data()
        
        status_text.text("🧠 Building content model...")
        progress_bar.progress(40)
        recommender.build_content_model()
        
        status_text.text("🔢 Creating user-item matrix...")
        progress_bar.progress(60)
        recommender.create_user_item_matrix()
        
        status_text.text("👥 Computing user similarity...")
        progress_bar.progress(75)
        recommender.compute_user_similarity()
        
        status_text.text("🎬 Computing item similarity...")
        progress_bar.progress(85)
        recommender.compute_item_similarity()
        
        if enable_svd:
            status_text.text("🤖 Training SVD model...")
            progress_bar.progress(95)
            recommender.train_svd()
        else:
            status_text.text("⏭️ Skipping SVD training...")
            progress_bar.progress(95)
        
        status_text.text("✅ Initialization complete!")
        progress_bar.progress(100)
        
        st.session_state.recommender = recommender
        st.session_state.is_initialized = True
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False

def display_movie_card(movie_row, score_column=None, show_overview=True):
    """Enhanced movie card with real posters from your data"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            poster_url = get_movie_poster(movie_row)
            
            with st.container():
                st.markdown('<div class="poster-container">', unsafe_allow_html=True)
                
                poster_image = load_poster_image(poster_url)
                
                if poster_image:
                    st.image(poster_image, width=150, caption=f"TMDb ID: {movie_row.get('tmdbId')}")
                else:
                    title = movie_row.get('title', 'Movie')[:12]
                    fallback_url = f"https://via.placeholder.com/300x450/667eea/white?text={title.replace(' ', '+')}"
                    st.image(fallback_url, width=150, caption=f"TMDb ID: {movie_row.get('tmdbId')}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"### 🎬 {movie_row.get('title', 'Unknown Title')}")
            
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.write(f"**🆔 TMDb ID:** `{movie_row.get('tmdbId', 'N/A')}`")
                
                if 'vote_average' in movie_row and pd.notna(movie_row['vote_average']):
                    rating = movie_row['vote_average']
                    stars = "⭐" * int(rating // 2) + "☆" * (5 - int(rating // 2))
                    st.write(f"**📊 Rating:** {stars} {rating:.1f}/10")
            
            with detail_col2:
                # Genres handling
                genres = None
                if 'genres' in movie_row and pd.notna(movie_row['genres']):
                    genres = movie_row['genres']
                elif 'genres_y' in movie_row and pd.notna(movie_row['genres_y']):
                    genres = movie_row['genres_y']
                
                if genres:
                    st.write(f"**🎭 Genres:** {genres}")
                
                # Poster info
                if 'poster_path' in movie_row and pd.notna(movie_row['poster_path']):
                    st.write(f"**🖼️ Poster:** ✅ Available")
                else:
                    st.write(f"**🖼️ Poster:** ❌ Not Available")
                
                # Score display
                if score_column and score_column in movie_row and pd.notna(movie_row[score_column]):
                    score_value = movie_row[score_column]
                    if score_column == 'similarity':
                        st.write(f"**🔗 Similarity:** `{score_value:.3f}`")
                        st.progress(float(score_value))
                    elif 'rating' in score_column:
                        st.write(f"**🎯 Predicted:** `{score_value:.2f}`")
                        st.progress(float(min(score_value/5, 1.0)))
                    elif 'score' in score_column:
                        st.write(f"**⚡ Hybrid:** `{score_value:.3f}`")
                        st.progress(float(score_value))
            
            # Overview
            if show_overview and 'overview' in movie_row and pd.notna(movie_row['overview']):
                st.write("**📖 Plot Overview:**")
                overview_text = movie_row['overview'][:300] + "..." if len(str(movie_row['overview'])) > 300 else movie_row['overview']
                st.markdown(f'<div class="overview-text">{overview_text}</div>', unsafe_allow_html=True)
        
        st.markdown("---")

def display_trending_card(movie_row, rank_info, icon, extra_info=None, compact=False):
    """Display a trending movie card with enhanced styling"""
    with st.container():
        if compact:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                poster_url = get_movie_poster(movie_row)
                try:
                    st.image(poster_url, width=80)
                except:
                    st.write("🖼️")
            
            with col2:
                st.markdown(f"**{icon} {movie_row.get('title', 'Unknown')}**")
                st.caption(f"{rank_info}")
                if extra_info:
                    st.caption(extra_info)
            
            with col3:
                if 'vote_average' in movie_row and pd.notna(movie_row['vote_average']):
                    rating = movie_row['vote_average']
                    st.metric("Rating", f"{rating:.1f}")
        else:
            card_col1, card_col2 = st.columns([1, 2])
            
            with card_col1:
                poster_url = get_movie_poster(movie_row)
                try:
                    st.image(poster_url, width=120)
                except:
                    st.image("https://via.placeholder.com/200x300/667eea/white?text=Movie+Poster", width=120)
            
            with card_col2:
                st.markdown(f"### {icon} {movie_row.get('title', 'Unknown Title')}")
                st.markdown(f"**{rank_info}**")
                
                if 'vote_average' in movie_row and pd.notna(movie_row['vote_average']):
                    rating = movie_row['vote_average']
                    stars = "⭐" * int(rating // 2) + "☆" * (5 - int(rating // 2))
                    st.write(f"**Rating:** {stars} {rating:.1f}/10")
                
                genres = None
                if 'genres' in movie_row and pd.notna(movie_row['genres']):
                    genres = movie_row['genres']
                elif 'genres_y' in movie_row and pd.notna(movie_row['genres_y']):
                    genres = movie_row['genres_y']
                
                if genres:
                    genre_list = genres.split(',')[:3]
                    st.write(f"**Genres:** {', '.join(genre_list)}")
                
                if extra_info:
                    st.info(extra_info)
                
                if 'overview' in movie_row and pd.notna(movie_row['overview']):
                    overview = str(movie_row['overview'])
                    short_overview = overview[:150] + "..." if len(overview) > 150 else overview
                    st.write(f"**Plot:** {short_overview}")
        
        st.markdown("---")

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Navigation")
        
        if not st.session_state.is_initialized:
            st.warning("⚠️ System not initialized")
            
            enable_svd = st.checkbox(
                "Enable SVD recommendations", 
                value=False,
                help="SVD provides more accurate recommendations but takes longer to initialize"
            )
            
            if st.button("🚀 Initialize System", type="primary"):
                with st.spinner("Initializing recommender system..."):
                    if initialize_recommender(enable_svd=enable_svd):
                        st.success("✅ System initialized successfully!")
                        st.rerun()
        else:
            st.success("✅ System Ready")
            
            # Navigation menu
            app_mode = st.selectbox(
                "Choose App Mode",
                ["🔥 Trending", "🔍 Search Movies", "🎯 Get Recommendations", "👤 Set User Profile", "📊 System Stats"],
                index=0
            )
            
            # Quick stats
            if st.session_state.recommender:
                recommender = st.session_state.recommender
                st.markdown("### 📈 Quick Stats")
                st.metric("Movies", f"{len(recommender.metadata_df):,}")
                st.metric("Users", f"{len(recommender.user_item_matrix.index):,}")
                st.metric("Ratings", f"{len(recommender.ratings_df):,}")
                
                # Show poster availability
                if 'poster_path' in recommender.metadata_df.columns:
                    poster_available = recommender.metadata_df['poster_path'].notna().sum()
                    st.metric("🖼️ Posters Available", f"{poster_available:,}")

    # Main content
    if not st.session_state.is_initialized:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ## 👋 Welcome!
            
            Please initialize the recommender system using the **Initialize System** button in the sidebar.
            
            ### 🚀 Enhanced Features:
            - **🖼️ Real Movie Posters**: Display actual movie posters from your data
            - **🧠 Smart Content Search**: Fuzzy matching for movie titles
            - **👥 Collaborative Filtering**: User and item-based recommendations  
            - **🤖 SVD**: Matrix factorization recommendations
            - **⚡ Hybrid**: Combined approach for best results
            - **🔍 Intelligent Search**: Auto-suggestions and partial matching
            """)
        return

    # Get app mode
    if 'app_mode' not in locals():
        app_mode = "🔥 Trending"
    
    recommender = st.session_state.recommender
    
    # Main pages
    if app_mode == "🔥 Trending":
        st.header("🔥 Trending Movies Dashboard")
        st.info("🎯 **Real-time Analytics**: Discover what's hot in movies right now!")
        
        # Quick stats with trending indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎬 Total Movies", 
                f"{len(recommender.metadata_df):,}",
                delta="Active Database"
            )
        with col2:
            st.metric(
                "👥 Active Users", 
                f"{len(recommender.user_item_matrix.index):,}",
                delta="Rating Contributors"
            )
        with col3:
            st.metric(
                "⭐ Total Ratings", 
                f"{len(recommender.ratings_df):,}",
                delta="User Interactions"
            )
        with col4:
            avg_rating = recommender.ratings_df['rating'].mean()
            st.metric(
                "📈 Avg Rating", 
                f"{avg_rating:.2f}",
                delta=f"Quality Score"
            )
        
        st.markdown("---")
        
        # Trending Categories Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔥 Top Rated", 
            "📈 Most Popular", 
            "⭐ Recently Added", 
            "🎭 By Genre", 
            "🏆 Awards Winners"
        ])
        
        with tab1:
            st.subheader("🔥 Highest Rated Movies")
            st.info("Movies with the highest average ratings from users")
            
            if 'vote_average' in recommender.metadata_df.columns:
                top_rated = recommender.metadata_df[
                    (recommender.metadata_df['vote_average'] >= 7.0) & 
                    (recommender.metadata_df['vote_average'].notna())
                ].nlargest(10, 'vote_average')
                
                col1, col2 = st.columns(2)
                for idx, (_, movie) in enumerate(top_rated.iterrows()):
                    if idx % 2 == 0:
                        with col1:
                            display_trending_card(movie, f"#{idx+1} Top Rated", "🔥")
                    else:
                        with col2:
                            display_trending_card(movie, f"#{idx+1} Top Rated", "🔥")
        
        with tab2:
            st.subheader("📈 Most Popular Movies")
            st.info("Movies with the most user ratings and interactions")
            
            movie_popularity = recommender.ratings_df['tmdbId'].value_counts().head(10)
            popular_movies = recommender.metadata_df[
                recommender.metadata_df['tmdbId'].isin(movie_popularity.index)
            ].copy()
            popular_movies['rating_count'] = popular_movies['tmdbId'].map(movie_popularity)
            popular_movies = popular_movies.sort_values('rating_count', ascending=False)
            
            col1, col2 = st.columns(2)
            for idx, (_, movie) in enumerate(popular_movies.iterrows()):
                if idx % 2 == 0:
                    with col1:
                        display_trending_card(
                            movie, 
                            f"#{idx+1} Most Popular", 
                            "📈",
                            extra_info=f"🎯 {movie['rating_count']} ratings"
                        )
                else:
                    with col2:
                        display_trending_card(
                            movie, 
                            f"#{idx+1} Most Popular", 
                            "📈",
                            extra_info=f"🎯 {movie['rating_count']} ratings"
                        )
        
        with tab3:
            st.subheader("⭐ Recently Trending")
            st.info("Movies gaining momentum in user ratings")
            
            if 'poster_path' in recommender.metadata_df.columns:
                recent_trending = recommender.metadata_df[
                    (recommender.metadata_df['poster_path'].notna()) &
                    (recommender.metadata_df['vote_average'] >= 6.5) &
                    (recommender.metadata_df['vote_average'].notna())
                ].sample(n=min(10, len(recommender.metadata_df)), random_state=42)
                
                col1, col2 = st.columns(2)
                for idx, (_, movie) in enumerate(recent_trending.iterrows()):
                    if idx % 2 == 0:
                        with col1:
                            display_trending_card(movie, f"#{idx+1} Trending Now", "⭐")
                    else:
                        with col2:
                            display_trending_card(movie, f"#{idx+1} Trending Now", "⭐")
        
        with tab4:
            st.subheader("🎭 Trending by Genre")
            st.info("Popular movies across different genres")
            
            if 'genres' in recommender.metadata_df.columns:
                movies_with_genres = recommender.metadata_df[
                    recommender.metadata_df['genres'].notna()
                ].copy()
                
                sample_genres = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Sci-Fi']
                
                for genre in sample_genres:
                    genre_movies = movies_with_genres[
                        movies_with_genres['genres'].str.contains(genre, case=False, na=False)
                    ]
                    
                    if not genre_movies.empty and 'vote_average' in genre_movies.columns:
                        top_genre_movie = genre_movies.nlargest(1, 'vote_average').iloc[0]
                        
                        st.markdown(f"### 🎭 **{genre}** - Trending")
                        display_trending_card(
                            top_genre_movie, 
                            f"Best in {genre}", 
                            "🎭",
                            compact=True
                        )
        
        with tab5:
            st.subheader("🏆 Award-Worthy Movies")
            st.info("Movies with exceptional ratings that deserve recognition")
            
            if 'vote_average' in recommender.metadata_df.columns:
                award_worthy = recommender.metadata_df[
                    (recommender.metadata_df['vote_average'] >= 8.0) & 
                    (recommender.metadata_df['vote_average'].notna())
                ].head(8)
                
                if not award_worthy.empty:
                    col1, col2 = st.columns(2)
                    for idx, (_, movie) in enumerate(award_worthy.iterrows()):
                        if idx % 2 == 0:
                            with col1:
                                display_trending_card(
                                    movie, 
                                    f"🏆 Award Worthy", 
                                    "🏆",
                                    extra_info=f"⭐ {movie['vote_average']:.1f}/10"
                                )
                        else:
                            with col2:
                                display_trending_card(
                                    movie, 
                                    f"🏆 Award Worthy", 
                                    "🏆",
                                    extra_info=f"⭐ {movie['vote_average']:.1f}/10"
                                )
        
        # Trending Analytics Section
        st.markdown("---")
        st.subheader("📊 Trending Analytics")
        
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            if 'vote_average' in recommender.metadata_df.columns:
                trending_ratings = recommender.metadata_df[
                    recommender.metadata_df['vote_average'] >= 7.0
                ]['vote_average']
                
                fig_trending = px.histogram(
                    x=trending_ratings,
                    nbins=20,
                    title="📈 Distribution of High-Rated Movies",
                    labels={'x': 'Rating', 'y': 'Number of Movies'},
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig_trending, use_container_width=True)
        
        with analytics_col2:
            if 'genres' in recommender.metadata_df.columns:
                all_genres = []
                for genres_str in recommender.metadata_df['genres'].dropna():
                    if isinstance(genres_str, str):
                        genres = [g.strip() for g in genres_str.split(',')]
                        all_genres.extend(genres)
                
                if all_genres:
                    genre_counts = pd.Series(all_genres).value_counts().head(10)
                    
                    fig_genres = px.bar(
                        x=genre_counts.values,
                        y=genre_counts.index,
                        orientation='h',
                        title="🎭 Most Popular Genres",
                        labels={'x': 'Number of Movies', 'y': 'Genre'},
                        color=genre_counts.values,
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_genres, use_container_width=True)
    
    elif app_mode == "🔍 Search Movies":
        st.header("🔍 Enhanced Movie Search with Real Posters")
        st.info("🧠 **Smart Search + Real Posters**: Type partial names, see actual movie posters!")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        with col1:
            search_query = st.text_input(
                "🔎 Enter movie title:", 
                placeholder="e.g., matr, incep, dark knight, aveng...",
                help="Partial matches work! Movies will show with real posters."
            )
        with col2:
            st.write("")
            if st.button("🗑️ Clear"):
                st.rerun()
        
        # Real-time search
        if search_query and len(search_query) >= 2:
            # Show suggestions first
            suggestions = recommender.get_movie_suggestions(search_query, 5)
            
            if suggestions:
                st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                st.write("💡 **Quick Suggestions:**")
                suggestion_cols = st.columns(min(len(suggestions), 3))
                for i, suggestion in enumerate(suggestions[:3]):
                    with suggestion_cols[i]:
                        if st.button(f"🎬 {suggestion}", key=f"suggest_{i}"):
                            search_query = suggestion
                            st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Perform enhanced search
            with st.spinner("🔍 Searching with AI-powered matching..."):
                search_results = recommender.search_movies_fuzzy(search_query, 20)
            
            if not search_results.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"🎉 Found {len(search_results)} matches")
                with col2:
                    max_results = st.slider("Show results:", 1, min(15, len(search_results)), 5)
                with col3:
                    sort_by = st.selectbox("Sort by:", ["Relevance", "Title A-Z", "Rating ⭐", "Has Poster"])
                
                # Sort results
                if sort_by == "Rating ⭐" and 'vote_average' in search_results.columns:
                    search_results = search_results.sort_values('vote_average', ascending=False, na_last=True)
                elif sort_by == "Title A-Z":
                    search_results = search_results.sort_values('title')
                elif sort_by == "Has Poster":
                    search_results = search_results.sort_values('poster_path', ascending=False, na_last=True)
                
                st.markdown("---")
                
                # Display results with real posters
                for idx, (_, movie) in enumerate(search_results.head(max_results).iterrows()):
                    st.markdown(f"### 🏆 #{idx + 1}")
                    display_movie_card(movie)
                    
            else:
                st.warning(f"🔍 No movies found for '{search_query}'")
    
    elif app_mode == "🎯 Get Recommendations":
        st.header("🎯 Enhanced Movie Recommendations with Real Posters")
        
        strategy = st.selectbox(
            "🧠 Choose Strategy:",
            ["hybrid", "content", "user", "item", "svd"],
            format_func=lambda x: {
                "content": "🧠 Content-Based (Smart Search)",
                "user": "👥 User-Based CF", 
                "item": "🎬 Item-Based CF",
                "svd": "🤖 SVD",
                "hybrid": "⚡ Hybrid (Best)"
            }[x]
        )
        
        if strategy == "content":
            st.subheader("🧠 Enhanced Content-Based Recommendations")
            st.info("🎯 **Smart Feature**: Type partial movie names + see real posters!")
            
            movie_input = st.text_input(
                "🎬 Enter a movie you like:", 
                placeholder="e.g., matr (Matrix), incep (Inception), dark (Dark Knight)...",
                help="Partial names work! See real movie posters in results."
            )
            
            # Show live suggestions as user types
            if movie_input and len(movie_input) >= 2:
                suggestions = recommender.get_movie_suggestions(movie_input, 5)
                if suggestions:
                    selected_movie = st.selectbox(
                        "💡 Or select from suggestions:",
                        options=[""] + suggestions,
                        format_func=lambda x: "Type manually or select..." if x == "" else f"🎬 {x}"
                    )
                    if selected_movie:
                        movie_input = selected_movie
            
            top_k = st.slider("Number of recommendations:", 1, 20, 10)
            
            if movie_input and st.button("🎯 Get Similar Movies", type="primary"):
                try:
                    with st.spinner(f"🤖 Finding movies similar to '{movie_input}' with real posters..."):
                        recommendations = recommender.recommend(
                            movie_title=movie_input,
                            strategy="content",
                            top_k=top_k
                        )
                    
                    if not recommendations.empty:
                        st.success(f"🎉 Found {len(recommendations)} movies similar to '{movie_input}'!")
                        
                        for idx, (_, movie) in enumerate(recommendations.iterrows()):
                            st.markdown(f"### 🎯 Recommendation #{idx + 1}")
                            display_movie_card(movie, 'similarity')
                    else:
                        st.warning("😔 No similar movies found.")
                        
                except ValueError as e:
                    error_msg = str(e)
                    if "Closest matches" in error_msg:
                        st.error(f"❌ {error_msg}")
                        st.info("💡 Try using the suggestions above or different keywords")
                    else:
                        st.error(f"❌ Error: {error_msg}")
        
        else:
            st.subheader("🎯 Personalized Recommendations with Real Posters")
            
            col1, col2 = st.columns(2)
            with col1:
                user_id = st.number_input("👤 User ID:", min_value=1, value=123, step=1)
            with col2:
                top_k = st.slider("📊 Number of recommendations:", 1, 20, 10)
            
            filter_seen = st.checkbox("🔍 Filter seen movies", value=True)
            
            if strategy == "hybrid":
                st.markdown("### ⚙️ Hybrid Strategy Weights")
                col1, col2, col3 = st.columns(3)
                with col1:
                    alpha = st.slider("👥 User weight:", 0.0, 1.0, 0.7, 0.1)
                with col2:
                    beta = st.slider("🎬 Item weight:", 0.0, 1.0, 0.3, 0.1)
                with col3:
                    delta = st.slider("💭 Sentiment weight:", 0.0, 1.0, 0.0, 0.1)
            
            if st.button("🎯 Get Recommendations", type="primary"):
                try:
                    with st.spinner(f"🤖 Generating {strategy} recommendations with real posters..."):
                        if strategy == "hybrid":
                            recommendations = recommender.recommend(
                                user_id=user_id,
                                strategy=strategy,
                                top_k=top_k,
                                filter_seen=filter_seen,
                                alpha=alpha,
                                beta=beta,
                                delta=delta
                            )
                        else:
                            recommendations = recommender.recommend(
                                user_id=user_id,
                                strategy=strategy,
                                top_k=top_k,
                                filter_seen=filter_seen
                            )
                    
                    if not recommendations.empty:
                        st.success(f"🎉 Top {len(recommendations)} recommendations for User {user_id}:")
                        
                        score_col = None
                        if 'predicted_rating' in recommendations.columns:
                            score_col = 'predicted_rating'
                        elif 'hybrid_score' in recommendations.columns:
                            score_col = 'hybrid_score'
                        
                        for idx, (_, movie) in enumerate(recommendations.iterrows()):
                            st.markdown(f"### 🎯 Recommendation #{idx + 1}")
                            display_movie_card(movie, score_col)
                    else:
                        st.warning("😔 No recommendations found.")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif app_mode == "👤 Set User Profile":
        st.header("👤 Enhanced User Profile Setup with Real Posters")
        
        user_id = st.number_input("🆔 New User ID:", min_value=1, value=999999, step=1)
        
        st.subheader("🎬 Smart Movie Selection with Real Posters")
        st.info("🧠 **Enhanced Search**: Use partial names to find movies with real posters!")
        
        search_term = st.text_input(
            "🔍 Search movies to add:", 
            placeholder="e.g., matr, incep, dark...",
            help="Partial names work! See real posters for selected movies"
        )
        
        if search_term and len(search_term) >= 2:
            search_results = recommender.search_movies_fuzzy(search_term, 10)
            
            if not search_results.empty:
                st.write("🔍 **Search Results with Real Posters:**")
                for _, movie in search_results.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                    with col1:
                        poster_url = get_movie_poster(movie)
                        try:
                            st.image(poster_url, width=80)
                        except:
                            st.write("🖼️")
                    with col2:
                        st.write(f"**{movie['title']}**")
                        st.caption(f"ID: {movie['tmdbId']}")
                        if 'genres' in movie and pd.notna(movie['genres']):
                            st.caption(f"🎭 {movie['genres'][:30]}...")
                    with col3:
                        if 'vote_average' in movie and pd.notna(movie['vote_average']):
                            st.caption(f"⭐ {movie['vote_average']:.1f}")
                    with col4:
                        if st.button("➕ Add", key=f"add_{movie['tmdbId']}"):
                            if movie['tmdbId'] not in st.session_state.selected_movies:
                                st.session_state.selected_movies.append(movie['tmdbId'])
                                st.success(f"Added '{movie['title']}'")
                                st.rerun()
                            else:
                                st.warning("Already added!")
        
        # Display selected movies
        if st.session_state.selected_movies:
            st.subheader(f"✅ Selected Movies with Posters ({len(st.session_state.selected_movies)})")
            
            selected_df = recommender.metadata_df[
                recommender.metadata_df['tmdbId'].isin(st.session_state.selected_movies)
            ]
            
            for _, movie in selected_df.iterrows():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    poster_url = get_movie_poster(movie)
                    try:
                        st.image(poster_url, width=100)
                    except:
                        st.write("🖼️ Poster")
                with col2:
                    st.write(f"**{movie['title']}**")
                    if 'vote_average' in movie and pd.notna(movie['vote_average']):
                        st.caption(f"⭐ {movie['vote_average']:.1f}")
                with col3:
                    if st.button("🗑️", key=f"remove_{movie['tmdbId']}", help="Remove"):
                        st.session_state.selected_movies.remove(movie['tmdbId'])
                        st.rerun()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🎯 Set Profile", type="primary"):
                    try:
                        recommender.set_user_profile(user_id, st.session_state.selected_movies)
                        st.success(f"✅ Profile created for User {user_id}!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            with col2:
                if st.button("🗑️ Clear All"):
                    st.session_state.selected_movies = []
                    st.rerun()
            with col3:
                st.info(f"Selected: {len(st.session_state.selected_movies)} movies")
    
    elif app_mode == "📊 System Stats":
        st.header("📊 Enhanced System Analytics with Poster Data")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎬 Movies", f"{len(recommender.metadata_df):,}")
        with col2:
            st.metric("👥 Users", f"{len(recommender.user_item_matrix.index):,}")
        with col3:
            st.metric("⭐ Ratings", f"{len(recommender.ratings_df):,}")
        with col4:
            avg_rating = recommender.ratings_df['rating'].mean()
            st.metric("📈 Avg Rating", f"{avg_rating:.2f}")
        
        # Poster statistics
        if 'poster_path' in recommender.metadata_df.columns:
            st.subheader("🖼️ Poster Data Analysis")
            col1, col2, col3 = st.columns(3)
            
            poster_available = recommender.metadata_df['poster_path'].notna().sum()
            poster_missing = len(recommender.metadata_df) - poster_available
            poster_percentage = (poster_available / len(recommender.metadata_df)) * 100
            
            with col1:
                st.metric("✅ Movies with Posters", f"{poster_available:,}")
            with col2:
                st.metric("❌ Movies without Posters", f"{poster_missing:,}")
            with col3:
                st.metric("📊 Poster Coverage", f"{poster_percentage:.1f}%")
        
        # Enhanced visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Rating Distribution", "🏆 Top Movies", "🖼️ Poster Analysis"])
        
        with tab1:
            st.subheader("📊 Rating Distribution")
            rating_counts = recommender.ratings_df['rating'].value_counts().sort_index()
            
            fig_ratings = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                title="Distribution of Movie Ratings",
                color=rating_counts.values,
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
        
        with tab2:
            if 'vote_average' in recommender.metadata_df.columns:
                st.subheader("🏆 Top Rated Movies with Posters")
                movies_with_posters = recommender.metadata_df[recommender.metadata_df['poster_path'].notna()]
                if not movies_with_posters.empty:
                    top_movies = movies_with_posters.nlargest(10, 'vote_average')[['title', 'vote_average']]
                    
                    fig_top = px.bar(
                        top_movies,
                        x='vote_average',
                        y='title',
                        orientation='h',
                        title="Top 10 Highest Rated Movies (with posters)",
                        color='vote_average',
                        color_continuous_scale="plasma"
                    )
                    st.plotly_chart(fig_top, use_container_width=True)
        
        with tab3:
            if 'poster_path' in recommender.metadata_df.columns:
                st.subheader("🖼️ Poster Availability Analysis")
                
                poster_data = pd.DataFrame({
                    'Status': ['Has Poster', 'No Poster'],
                    'Count': [poster_available, poster_missing]
                })
                
                fig_poster = px.pie(
                    poster_data,
                    values='Count',
                    names='Status',
                    title="Poster Data Availability",
                    color_discrete_map={'Has Poster': '#28a745', 'No Poster': '#dc3545'}
                )
                st.plotly_chart(fig_poster, use_container_width=True)

if __name__ == "__main__":
    main()

