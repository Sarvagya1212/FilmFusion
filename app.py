import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from datetime import datetime
import time

try:
    # Add the project root to Python path
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import the recommender system
    from src.recommenders.recommender_system import RecommenderSystem
    
except ImportError as e:
    st.error(f"❌ Import Error: {str(e)}")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="FilmFusion Pro: Advanced Movie Recommender", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .movie-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .metric-card {
        background: linear-gradient(45deg, #FA8072, #FFB347);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: white;
        margin: 0.5rem;
    }
    .genre-tag {
        background: linear-gradient(45deg, #FF8A80, #FF5722);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 2px;
        display: inline-block;
    }
    .sentiment-positive { color: #4CAF50; font-weight: bold; }
    .sentiment-negative { color: #F44336; font-weight: bold; }
    .sentiment-neutral { color: #FF9800; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Caching the Recommender
@st.cache_resource
def initialize_recommender():
    """Load and initialize the complete recommender system."""
    current_dir = os.path.dirname(__file__)
    ratings_path = os.path.join(current_dir, 'data', 'ratings_cleans.csv')
    metadata_path = os.path.join(current_dir, 'data', 'movies_with_sentiment.csv')
    
    if not os.path.exists(ratings_path) or not os.path.exists(metadata_path):
        st.error("Data files not found. Please ensure the data files exist.")
        st.info("Expected files:")
        st.code(f"- {ratings_path}\n- {metadata_path}")
        return None
    
    try:
        recommender = RecommenderSystem(
            ratings_path=ratings_path, 
            metadata_path=metadata_path, 
            enable_rl=True
        )
        # Disable evaluation initially to avoid startup errors
        recommender.initialize_all(evaluate_models=False)
        
        # Pre-compute most common user similarity calculations for faster lookup
        if hasattr(recommender, 'user_similarity_df') and recommender.user_similarity_df is not None:
            # Cache top 10 similar users for each user for faster lookup
            recommender._cached_similar_users = {}
            user_ids = list(recommender.user_profiles.keys())[:100] if hasattr(recommender, 'user_profiles') else []
            
            for user_id in user_ids:
                try:
                    if user_id in recommender.user_similarity_df.index:
                        similar_users = recommender.user_similarity_df[user_id].sort_values(ascending=False)[1:11]
                        recommender._cached_similar_users[user_id] = similar_users.to_dict()
                except Exception as e:
                    # Skip users that cause errors in similarity calculation
                    continue
        
        recommender._precompute_popular_movies()
        recommender._precompute_genre_statistics()
        recommender._precompute_user_clusters()
        return recommender
        
    except Exception as e:
        st.error(f"Error initializing recommender: {str(e)}")
        return None


# Update your app.py with lazy loading
def create_lazy_tabs():
    """Create tabs with lazy loading"""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Recommendations", 
        "🔍 Similar Movies", 
        "📊 Analytics",
        "🤖 RL Dashboard",
        "📈 Data Explorer"
    ])
    
    # Only load active tab content
    return tab1, tab2, tab3, tab4, tab5

# In your main app, use session state to track loaded content
if 'loaded_tabs' not in st.session_state:
    st.session_state.loaded_tabs = set()

# Load tab content only when accessed
def load_tab_content(tab_name, content_function):
    if tab_name not in st.session_state.loaded_tabs:
        with st.spinner(f"Loading {tab_name}..."):
            content_function()
            st.session_state.loaded_tabs.add(tab_name)
    else:
        content_function()


# Enhanced UI Helper Functions
def display_movie_card(movie, score_col=None, rank=None):
    """Enhanced movie card display with all features"""
    rank_str = f"#{rank} " if rank else ""
    
    # Extract year from release_date
    year = extract_year_from_date(movie.get('release_date', ''))
    
    st.markdown(f"""
    <div class="movie-card">
        <h3>🎬 {rank_str}{movie.get('title', 'Unknown Title')} ({year})</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster_url = f"https://image.tmdb.org/t/p/w300{movie.get('poster_path', '')}"
        try:
            if movie.get('poster_path'):
                st.image(poster_url, width=150)
            else:
                st.markdown("🎬", unsafe_allow_html=True)
        except:
            st.markdown("🎬", unsafe_allow_html=True)
    
    with col2:
        # Score Display
        if score_col and score_col in movie and not pd.isna(movie[score_col]):
            score = movie[score_col]
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Recommendation Score</h4>
                <h2>{score:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Movie Details in colorful format
        col2a, col2b = st.columns(2)
        
        with col2a:
            # Genres with colorful tags
            genres_str = movie.get('genres', 'N/A')
            if genres_str != 'N/A' and not pd.isna(genres_str):
                try:
                    # Parse genres list
                    import ast
                    genres = ast.literal_eval(genres_str) if isinstance(genres_str, str) else genres_str
                    genres_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in genres[:4]])
                    st.markdown(f"**🎭 Genres:** {genres_html}", unsafe_allow_html=True)
                except:
                    st.markdown(f"**🎭 Genres:** {genres_str}")
            else:
                st.markdown("**🎭 Genres:** N/A")
            
            # Ratings
            if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                vote_avg = movie['vote_average']
                st.markdown(f"**⭐ TMDB Rating:** {vote_avg}/10")
            
            if 'avg_ratings' in movie and not pd.isna(movie['avg_ratings']):
                local_avg = movie['avg_ratings']
                st.markdown(f"**📈 Avg Rating:** {local_avg:.2f}/5")
            
            # Popularity
            if 'popularity' in movie and not pd.isna(movie['popularity']):
                popularity = movie['popularity']
                st.markdown(f"**🔥 Popularity:** {popularity:.1f}")
        
        with col2b:
            # Release Info
            st.markdown(f"**📅 Release Date:** {movie.get('release_date', 'N/A')}")
            
            # Runtime
            if 'runtime' in movie and not pd.isna(movie['runtime']):
                runtime = int(movie['runtime'])
                hours = runtime // 60
                minutes = runtime % 60
                st.markdown(f"**⏱️ Runtime:** {hours}h {minutes}m")
            
            # Language
            if 'language' in movie and not pd.isna(movie['language']):
                st.markdown(f"**🌍 Language:** {movie['language'].upper()}")
            
            # Vote Count
            if 'vote_count' in movie and not pd.isna(movie['vote_count']):
                votes = int(movie['vote_count'])
                st.markdown(f"**👥 Votes:** {votes:,}")
        
        # Cast Information
        if 'cast' in movie and not pd.isna(movie['cast']):
            cast_str = str(movie['cast'])
            if len(cast_str) > 100:
                cast_str = cast_str[:100] + "..."
            st.markdown(f"**🎭 Cast:** {cast_str}")
        
        # Sentiment Analysis
        if 'avg_sentiment' in movie and not pd.isna(movie['avg_sentiment']):
            sentiment_score = movie['avg_sentiment']
            sentiment_label, sentiment_class = get_sentiment_label(sentiment_score)
            st.markdown(f"""**💭 Sentiment:** <span class="{sentiment_class}">{sentiment_label} ({sentiment_score:.2f})</span>""", 
                       unsafe_allow_html=True)
        
        # Overview in expandable section
        with st.expander("📝 Plot Overview & Details"):
            overview = movie.get('overview', 'No overview available.')
            if pd.isna(overview) or overview == '':
                overview = 'No overview available.'
            st.markdown(f"**Plot:** {overview}")
            
            if 'tagline' in movie and not pd.isna(movie['tagline']) and movie['tagline'] != '':
                st.markdown(f"**💡 Tagline:** *{movie['tagline']}*")
            
            if 'keywords' in movie and not pd.isna(movie['keywords']):
                st.markdown(f"**🔑 Keywords:** {movie['keywords']}")


def extract_year_from_date(date_str):
    """Extract year from release_date string"""
    try:
        if pd.isna(date_str):
            return "N/A"
        return str(pd.to_datetime(date_str).year)
    except:
        return "N/A"

def get_sentiment_label(sentiment_score):
    """Convert sentiment score to label and color"""
    if pd.isna(sentiment_score):
        return "Unknown", "sentiment-neutral"
    elif sentiment_score > 0.6:
        return "Very Positive", "sentiment-positive"
    elif sentiment_score > 0.3:
        return "Positive", "sentiment-positive"
    elif sentiment_score > -0.1:
        return "Neutral", "sentiment-neutral"
    elif sentiment_score > -0.3:
        return "Negative", "sentiment-negative"
    else:
        return "Very Negative", "sentiment-negative"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_recommendations(user_id, strategy, top_k):
    """Cache recommendations to avoid recomputation"""
    try:
        return recommender.recommend(
            user_id=user_id, 
            strategy=strategy, 
            top_k=top_k
        )
    except Exception as e:
        st.error(f"Error with {strategy}: {str(e)}")
        return pd.DataFrame()


def display_enhanced_movie_card(movie, score_col=None, rank=None):
    """Enhanced movie card display with all features"""
    rank_str = f"#{rank} " if rank else ""
    
    # Extract year from release_date
    year = extract_year_from_date(movie.get('release_date', ''))
    
    st.markdown(f"""
    <div class="movie-card">
        <h3>🎬 {rank_str}{movie.get('title', 'Unknown Title')} ({year})</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        poster_url = f"https://image.tmdb.org/t/p/w300{movie.get('poster_path', '')}"
        try:
            if movie.get('poster_path'):
                st.image(poster_url, width=150)
            else:
                st.markdown("🎬", unsafe_allow_html=True)
        except:
            st.markdown("🎬", unsafe_allow_html=True)
    
    with col2:
        # Score Display
        if score_col and score_col in movie and not pd.isna(movie[score_col]):
            score = movie[score_col]
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Recommendation Score</h4>
                <h2>{score:.3f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Movie Details in colorful format
        col2a, col2b = st.columns(2)
        
        with col2a:
            # Genres with colorful tags
            genres_str = movie.get('genres', 'N/A')
            if genres_str != 'N/A' and not pd.isna(genres_str):
                try:
                    # Parse genres list
                    import ast
                    genres = ast.literal_eval(genres_str) if isinstance(genres_str, str) else genres_str
                    genres_html = "".join([f'<span class="genre-tag">{genre}</span>' for genre in genres[:4]])
                    st.markdown(f"**🎭 Genres:** {genres_html}", unsafe_allow_html=True)
                except:
                    st.markdown(f"**🎭 Genres:** {genres_str}")
            else:
                st.markdown("**🎭 Genres:** N/A")
            
            # Ratings
            if 'vote_average' in movie and not pd.isna(movie['vote_average']):
                vote_avg = movie['vote_average']
                st.markdown(f"**⭐ TMDB Rating:** {vote_avg}/10")
            
            if 'avg_ratings' in movie and not pd.isna(movie['avg_ratings']):
                local_avg = movie['avg_ratings']
                st.markdown(f"**📈 Avg Rating:** {local_avg:.2f}/5")
            
            # Popularity
            if 'popularity' in movie and not pd.isna(movie['popularity']):
                popularity = movie['popularity']
                st.markdown(f"**🔥 Popularity:** {popularity:.1f}")
        
        with col2b:
            # Release Info
            st.markdown(f"**📅 Release Date:** {movie.get('release_date', 'N/A')}")
            
            # Runtime
            if 'runtime' in movie and not pd.isna(movie['runtime']):
                runtime = int(movie['runtime'])
                hours = runtime // 60
                minutes = runtime % 60
                st.markdown(f"**⏱️ Runtime:** {hours}h {minutes}m")
            
            # Language
            if 'language' in movie and not pd.isna(movie['language']):
                st.markdown(f"**🌍 Language:** {movie['language'].upper()}")
            
            # Vote Count
            if 'vote_count' in movie and not pd.isna(movie['vote_count']):
                votes = int(movie['vote_count'])
                st.markdown(f"**👥 Votes:** {votes:,}")
        
        # Cast Information
        if 'cast' in movie and not pd.isna(movie['cast']):
            cast_str = str(movie['cast'])
            if len(cast_str) > 100:
                cast_str = cast_str[:100] + "..."
            st.markdown(f"**🎭 Cast:** {cast_str}")
        
        # Sentiment Analysis
        if 'avg_sentiment' in movie and not pd.isna(movie['avg_sentiment']):
            sentiment_score = movie['avg_sentiment']
            sentiment_label, sentiment_class = get_sentiment_label(sentiment_score)
            st.markdown(f"""**💭 Sentiment:** <span class="{sentiment_class}">{sentiment_label} ({sentiment_score:.2f})</span>""", 
                       unsafe_allow_html=True)
        
        # Overview in expandable section
        with st.expander("📝 Plot Overview & Details"):
            overview = movie.get('overview', 'No overview available.')
            if pd.isna(overview) or overview == '':
                overview = 'No overview available.'
            st.markdown(f"**Plot:** {overview}")
            
            if 'tagline' in movie and not pd.isna(movie['tagline']) and movie['tagline'] != '':
                st.markdown(f"**💡 Tagline:** *{movie['tagline']}*")
            
            if 'keywords' in movie and not pd.isna(movie['keywords']):
                st.markdown(f"**🔑 Keywords:** {movie['keywords']}")

def safe_get_user_profiles(recommender):
    """Safely get user profiles, handling potential initialization issues."""
    try:
        if hasattr(recommender, 'user_profiles') and recommender.user_profiles:
            return list(recommender.user_profiles.keys())
        elif hasattr(recommender, 'ratings_df') and recommender.ratings_df is not None:
            return sorted(recommender.ratings_df['userId'].unique())
        else:
            return []
    except Exception:
        return []

# Main Application
st.markdown('<h1 class="main-header">🎬 FilmFusion Pro: Advanced Movie Recommender</h1>', unsafe_allow_html=True)
st.markdown("*Powered by Neural Collaborative Filtering & Reinforcement Learning with Sentiment Analysis*")

# Initialize System
with st.spinner("🚀 Initializing advanced recommendation engines..."):
    recommender = initialize_recommender()

if not recommender:
    st.error("❌ Failed to initialize the recommender system. Please check your data files and try again.")
    st.stop()

# Display initialization success with colorful metrics
st.success("✅ Recommender system initialized successfully!")

# Add this BEFORE the tab creation (after the recommender initialization)

# Performance Mode Toggle (place this before creating tabs)
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance Settings")

use_fast_mode = st.sidebar.radio(
    "Choose Mode:",
    [True, False],
    format_func=lambda x: "🚀 Fast Mode" if x else "🎯 Accurate Mode",
    help="Fast mode uses simplified algorithms for quick results",
    key="performance_mode"
)

if use_fast_mode:
    st.sidebar.info("Using optimized algorithms for faster responses")
else:
    st.sidebar.info("Using full algorithms for maximum accuracy")

# Add the cached function at the top level (after imports, before main app)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_recommendations(user_id, strategy, top_k):
    """Cache recommendations to avoid recomputation"""
    try:
        return recommender.recommend(
            user_id=user_id, 
            strategy=strategy, 
            top_k=top_k
        )
    except Exception as e:
        st.error(f"Error with {strategy}: {str(e)}")
        return pd.DataFrame()


# Create Tabs with colorful icons
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Get Recommendations", 
    "🔍 Find Similar Movies", 
    "📊 Model Performance",
    "🤖 RL Analytics",
    "📈 Data Insights",
    "🎭 Movie Explorer"
])

# Tab 1: Enhanced Recommendations
# In Tab 1: Enhanced Recommendations
with tab1:
    st.header("🎯 Advanced Personalized Recommendations")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        user_ids = safe_get_user_profiles(recommender)
        if user_ids:
            selected_user = st.selectbox("👤 Select User ID:", user_ids[:100])
        else:
            st.error("No user profiles available.")
            selected_user = None
    
    with col2:
        # Simplified strategy options for faster switching
        strategy_options = {
            "👥 User-Based CF": "user_cf",
            "🎬 Item-Based CF": "item_cf", 
            "🔀 Hybrid Approach": "hybrid",
            # Keep advanced algorithms but warn about performance
            "🧠 Advanced Neural CF (Slower)": "advanced_ncf",
            "🎰 RL: ε-Greedy (Slower)": "rl_epsilon"
        }
        selected_strategy_display = st.selectbox("🤖 Algorithm:", list(strategy_options.keys()))
        strategy_key = strategy_options[selected_strategy_display]

    # Number of recommendations slider
    num_recs = st.slider("📊 Number of Recommendations:", 3, 10, 5)  # Reduced max

    # In Tab 1: Enhanced Recommendations
if st.button("🚀 Generate Recommendations", type="primary"):
    if selected_user:
        start_time = time.time()
        
        # Choose recommendation method based on performance mode
        if use_fast_mode:  # Now properly defined
            with st.spinner(f"⚡ Fast generating {selected_strategy_display}..."):
                try:
                    # For fast mode, use simpler algorithms
                    if strategy_key in ['advanced_ncf', 'rl_epsilon', 'rl_ucb', 'rl_thompson']:
                        # Use hybrid as fast alternative for complex algorithms
                        recommendations = recommender.recommend_hybrid(selected_user, num_recs)
                    else:
                        recommendations = recommender.recommend(
                            user_id=selected_user, 
                            strategy=strategy_key, 
                            top_k=num_recs
                        )
                except Exception as e:
                    st.error(f"Fast mode error: {str(e)}")
                    recommendations = recommender.recommend_hybrid(selected_user, num_recs)
        else:
            # Use caching for slower methods
            with st.spinner(f"🎬 Generating {selected_strategy_display} recommendations..."):
                recommendations = get_cached_recommendations(
                    selected_user, strategy_key, num_recs
                )
        
        end_time = time.time()
        
        # Display results with timing
        if not recommendations.empty:
            mode_indicator = "⚡ Fast" if use_fast_mode else "🎯 Accurate"
            st.success(f"✨ **Top {num_recs} Recommendations for User {selected_user}** | {mode_indicator} Mode ⏱️ `{end_time-start_time:.2f}s`")
            st.markdown(f"*🤖 Algorithm: {selected_strategy_display}*")
            
            # Determine score column
            score_cols = ['hybrid_score', 'predicted_rating', 'ncf_score', 'rl_score', 'similarity_score']
            score_col = None
            for col in score_cols:
                if col in recommendations.columns:
                    score_col = col
                    break
            
            for i, (_, movie) in enumerate(recommendations.head(num_recs).iterrows()):
                display_enhanced_movie_card(movie, score_col, rank=i+1)
                st.markdown("---")
        else:
            st.warning("⚠️ Could not generate recommendations. Try a different algorithm.")
    else:
        st.error("Please select a user ID.")


# Tab 2: Enhanced Similar Movies
with tab2:
    st.header("🔍 Find Movies Similar to Your Favorites")
    
    if hasattr(recommender, 'metadata_df') and recommender.metadata_df is not None:
        # Movie search with more features
        col1, col2 = st.columns([2, 1])
        with col1:
            movie_list = recommender.metadata_df['title'].dropna().tolist()
            selected_movie = st.selectbox("🎬 Select a movie:", movie_list[:500], key="similar_movie_select")
        
        with col2:
            num_similar = st.slider("📊 Number of Similar Movies:", 3, 15, 5)
        
        if st.button("🔍 Find Similar Movies", key="find_similar"):
            with st.spinner(f"🎬 Finding movies similar to '{selected_movie}'..."):
                try:
                    similar_movies = recommender.recommend(
                        movie_title=selected_movie, 
                        strategy='content', 
                        top_k=num_similar
                    )
                    
                    if not similar_movies.empty:
                        st.success(f"🎯 **Movies Similar to '{selected_movie}'**")
                        
                        for i, (_, movie) in enumerate(similar_movies.iterrows()):
                            display_enhanced_movie_card(movie, 'similarity_score', rank=i+1)
                            st.markdown("---")
                    else:
                        st.warning("Could not find similar movies.")
                        st.info("This might be due to the movie not being in our content database.")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.error("Movie database not available.")

# Tab 3: Model Performance (same as before but with colorful styling)
with tab3:
    st.header("📊 Model Performance Evaluation")
    st.markdown("Comprehensive evaluation using Precision@K, Recall@K, and nDCG@K metrics")
    
    # Add evaluation control
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("📊 Model evaluation is disabled by default for faster startup. Click the button to run evaluation.")
    with col2:
        if st.button("🔬 Run Evaluation"):
            with st.spinner("⏳ Running comprehensive evaluation... This may take a few minutes."):
                try:
                    recommender.evaluate_all_strategies()
                    st.success("✅ Evaluation completed!")
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
    
    # Check if evaluation results exist and are valid
    if (hasattr(recommender, 'evaluation_results') and 
        recommender.evaluation_results and 
        isinstance(recommender.evaluation_results, dict) and 
        "error" not in recommender.evaluation_results):
        
        try:
            # Validate results structure
            valid_results = {}
            for strategy, metrics in recommender.evaluation_results.items():
                if isinstance(metrics, dict) and len(metrics) > 0:
                    valid_results[strategy] = metrics
            
            if not valid_results:
                st.warning("⚠️ No valid evaluation results available.")
            else:
                # Create DataFrame safely with proper column handling
                eval_df = pd.DataFrame(valid_results).T
                
                # Check the actual shape and columns
                st.write(f"Debug: DataFrame shape: {eval_df.shape}")
                st.write(f"Debug: DataFrame columns: {list(eval_df.columns)}")
                
                # Reset index and handle column assignment properly
                eval_df = eval_df.reset_index()
                
                # Determine actual columns based on what we have
                actual_columns = ['Algorithm'] + list(eval_df.columns[1:])
                
                # Expected metric columns
                expected_metrics = ['Precision@K', 'Recall@K', 'nDCG@K', 'Coverage', 'Successful_Evaluations']
                
                # Map actual columns to expected names
                if len(eval_df.columns) == 6:  # Algorithm + 5 metrics
                    eval_df.columns = ['Algorithm'] + expected_metrics
                elif len(eval_df.columns) == 5:  # Algorithm + 4 metrics  
                    eval_df.columns = ['Algorithm'] + expected_metrics[:4]
                else:
                    # Fallback: use actual column structure
                    eval_df.columns = actual_columns
                
                # Ensure all metric columns are numeric
                metric_cols = [col for col in eval_df.columns if col != 'Algorithm']
                for col in metric_cols:
                    if col in eval_df.columns:
                        eval_df[col] = pd.to_numeric(eval_df[col], errors='coerce').fillna(0.0)
                
                # Display the DataFrame for debugging
                st.subheader("📋 Evaluation Results")
                st.dataframe(eval_df, use_container_width=True)
                
                # Key Metrics Display (only if we have the expected columns)
                if all(col in eval_df.columns for col in ['Precision@K', 'Recall@K', 'nDCG@K', 'Coverage']):
                    st.subheader("🏆 Performance Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    if len(eval_df) > 0:
                        best_precision = eval_df.loc[eval_df['Precision@K'].idxmax()]
                        best_recall = eval_df.loc[eval_df['Recall@K'].idxmax()]
                        best_ndcg = eval_df.loc[eval_df['nDCG@K'].idxmax()]
                        best_coverage = eval_df.loc[eval_df['Coverage'].idxmax()]
                        
                        col1.metric("🎯 Best Precision", f"{best_precision['Precision@K']:.3f}", best_precision['Algorithm'])
                        col2.metric("📈 Best Recall", f"{best_recall['Recall@K']:.3f}", best_recall['Algorithm'])
                        col3.metric("🔝 Best nDCG", f"{best_ndcg['nDCG@K']:.3f}", best_ndcg['Algorithm'])
                        col4.metric("📊 Best Coverage", f"{best_coverage['Coverage']:.3f}", best_coverage['Algorithm'])
                        
                        # Performance Visualization
                        st.subheader("📊 Performance Comparison")
                        
                        # Create radar chart
                        metrics = ['Precision@K', 'Recall@K', 'nDCG@K', 'Coverage']
                        
                        fig = go.Figure()
                        
                        for _, row in eval_df.iterrows():
                            fig.add_trace(go.Scatterpolar(
                                r=[row[metric] for metric in metrics if metric in row],
                                theta=[metric for metric in metrics if metric in row],
                                fill='toself',
                                name=row['Algorithm']
                            ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True,
                            title="Algorithm Performance Comparison"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                # Metric Explanations
                with st.expander("📖 Understanding the Metrics"):
                    st.markdown("""
                    **Precision@K:** Of the K movies recommended, what fraction did the user actually like?
                    - Higher is better (max = 1.0)
                    - Measures recommendation accuracy
                    
                    **Recall@K:** Of all movies the user liked, what fraction did we successfully recommend?
                    - Higher is better (max = 1.0)  
                    - Measures recommendation completeness
                    
                    **nDCG@K:** How well did we rank the recommendations (putting best items first)?
                    - Higher is better (max = 1.0)
                    - The most important metric for ranking quality
                    
                    **Coverage:** What fraction of test users received successful recommendations?
                    - Higher is better (max = 1.0)
                    - Measures algorithm robustness
                    """)
                    
        except Exception as e:
            st.error(f"❌ Error processing evaluation results: {str(e)}")
            st.info("💡 Debug information:")
            if hasattr(recommender, 'evaluation_results'):
                st.json(recommender.evaluation_results)
            
    elif hasattr(recommender, 'evaluation_results') and "error" in recommender.evaluation_results:
        st.error(f"❌ Evaluation Error: {recommender.evaluation_results['error']}")
        st.info("💡 The evaluation failed. This is often due to insufficient data or data quality issues.")
    else:
        st.warning("⚠️ Model evaluation not available. Click 'Run Evaluation' to generate performance metrics.")
        st.info("💡 The recommender system is still functional for generating recommendations.")

# Tab 4: RL Analytics (keeping existing code but with enhanced styling)
with tab4:
    st.header("🤖 Reinforcement Learning Analytics")
    
    if hasattr(recommender, 'get_bandit_analytics'):
        try:
            bandit_analytics = recommender.get_bandit_analytics()
            
            if bandit_analytics and "error" not in bandit_analytics:
                # Global Bandit Stats
                st.subheader("🌐 Global Bandit Performance")
                global_stats = bandit_analytics.get("global_bandit", {})
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Arms", f"{global_stats.get('total_arms', 0):,}")
                col2.metric("Total Pulls", f"{global_stats.get('total_pulls', 0):,}")  
                col3.metric("Avg Reward", f"{global_stats.get('avg_reward', 0):.3f}")
                
                # User Bandits Stats
                st.subheader("👥 User Bandits Overview")
                user_stats = bandit_analytics.get("user_bandits", {})
                
                col1, col2 = st.columns(2)
                col1.metric("Total Users", f"{user_stats.get('total_users', 0):,}")
                col2.metric("Total User Pulls", f"{user_stats.get('total_user_pulls', 0):,}")
                
                # Feedback History
                st.subheader("📈 Recent Feedback")
                feedback_stats = bandit_analytics.get("feedback_history", {})
                
                col1, col2 = st.columns(2)
                col1.metric("Total Feedback", f"{feedback_stats.get('total_feedback', 0):,}")
                col2.metric("Recent Avg Rating", f"{feedback_stats.get('recent_avg_rating', 0):.2f}")
                
                # Best Performing Movies
                if global_stats.get('best_arm'):
                    st.subheader("🏆 Top Performing Movie (Global)")
                    best_movie_id = global_stats['best_arm']
                    best_movie = recommender.metadata_df[recommender.metadata_df['tmdbId'] == best_movie_id]
                    if not best_movie.empty:
                        display_movie_card(best_movie.iloc[0])
                
                # RL Algorithm Comparison
                if hasattr(recommender, 'algorithm_performance'):
                    st.subheader("🔬 RL Algorithm Performance")
                    perf_data = recommender.algorithm_performance
                    
                    if any(perf['count'] > 0 for perf in perf_data.values()):
                        perf_df = pd.DataFrame(perf_data).T
                        st.dataframe(perf_df, use_container_width=True)
                    else:
                        st.info("No RL algorithm performance data available yet.")
                        
            else:
                error_msg = bandit_analytics.get('error', 'Unknown error') if bandit_analytics else 'RL not initialized'
                st.error(f"❌ RL Analytics not available: {error_msg}")
                
        except Exception as e:
            st.error(f"❌ Error retrieving RL analytics: {str(e)}")
    else:
        st.warning("⚠️ Reinforcement Learning analytics not available.")
        st.info("RL features may be disabled or not properly initialized.")


# Tab 5: Enhanced Data Insights
with tab5:
    st.header("📈 Enhanced Data Insights & Statistics")
    
    # System Overview with colorful metrics
    st.subheader("🎯 System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎬 Total Movies</h4>
                <h2>{len(recommender.metadata_df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⭐ Total Ratings</h4>
                <h2>{len(recommender.ratings_df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            user_count = len(recommender.user_profiles) if hasattr(recommender, 'user_profiles') else len(recommender.ratings_df['userId'].unique())
            st.markdown(f"""
            <div class="metric-card">
                <h4>👥 Total Users</h4>
                <h2>{user_count:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_rating = recommender.ratings_df['rating'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Avg Rating</h4>
                <h2>{avg_rating:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error displaying system overview: {str(e)}")
    
    # Enhanced Visualizations
    st.subheader("📊 Enhanced Data Visualizations")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        # Rating Distribution with enhanced styling
        st.markdown("**⭐ Rating Distribution**")
        try:
            rating_counts = recommender.ratings_df['rating'].value_counts().sort_index()
            fig_ratings = px.bar(
                x=rating_counts.index, 
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                title="Distribution of User Ratings",
                color=rating_counts.values,
                color_continuous_scale='viridis'
            )
            fig_ratings.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_ratings, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating rating distribution: {str(e)}")
    
    with fig_col2:
        # Sentiment Distribution
        st.markdown("**💭 Sentiment Analysis Distribution**")
        try:
            sentiment_data = recommender.metadata_df['avg_sentiment'].dropna()
            if len(sentiment_data) > 0:
                fig_sentiment = px.histogram(
                    x=sentiment_data,
                    nbins=30,
                    labels={'x': 'Sentiment Score', 'y': 'Count'},
                    title="Movie Sentiment Distribution",
                    color_discrete_sequence=['#FF6B6B']
                )
                fig_sentiment.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("Sentiment data not available.")
        except Exception as e:
            st.error(f"Error creating sentiment distribution: {str(e)}")
    
    # Language Distribution
    st.subheader("🌍 Language & Release Year Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Language distribution
        try:
            if 'language' in recommender.metadata_df.columns:
                lang_counts = recommender.metadata_df['language'].value_counts().head(10)
                fig_lang = px.pie(
                    values=lang_counts.values,
                    names=lang_counts.index,
                    title="Top 10 Languages",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_lang, use_container_width=True)
            else:
                st.info("Language data not available")
        except Exception as e:
            st.error(f"Error creating language chart: {str(e)}")
    
    with col2:
        # Release year analysis
        try:
            if 'release_date' in recommender.metadata_df.columns:
                # Extract years from release_date
                years = pd.to_datetime(recommender.metadata_df['release_date'], errors='coerce').dt.year
                year_counts = years.value_counts().sort_index()
                # Filter to reasonable range
                year_counts = year_counts[(year_counts.index >= 1980) & (year_counts.index <= 2025)]
                
                fig_years = px.line(
                    x=year_counts.index,
                    y=year_counts.values,
                    title="Movies by Release Year (1980-2025)",
                    labels={'x': 'Year', 'y': 'Number of Movies'}
                )
                fig_years.update_traces(line_color='#4ECDC4', line_width=3)
                st.plotly_chart(fig_years, use_container_width=True)
            else:
                st.info("Release date data not available")
        except Exception as e:
            st.error(f"Error creating year analysis: {str(e)}")

# Tab 6: Movie Explorer (New Tab)
with tab6:
    st.header("🎭 Interactive Movie Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Year filter
        min_year = st.number_input("📅 Minimum Year:", 1980, 2025, 2000)
        max_year = st.number_input("📅 Maximum Year:", min_year, 2025, 2025)
    
    with col2:
        # Rating filter
        min_rating = st.slider("⭐ Minimum Rating:", 0.0, 10.0, 6.0)
        min_votes = st.number_input("👥 Minimum Votes:", 0, 50000, 100)
    
    with col3:
        # Genre filter
        if hasattr(recommender, 'metadata_df'):
            # Extract all unique genres
            all_genres = set()
            for genres_str in recommender.metadata_df['genres'].dropna():
                try:
                    import ast
                    genres = ast.literal_eval(genres_str) if isinstance(genres_str, str) else genres_str
                    all_genres.update(genres)
                except:
                    pass
            
            selected_genres = st.multiselect("🎭 Filter by Genres:", sorted(all_genres))
    
    if st.button("🔍 Explore Movies"):
        try:
            # Filter movies based on criteria
            filtered_df = recommender.metadata_df.copy()
            
            # Year filter
            filtered_df['year'] = pd.to_datetime(filtered_df['release_date'], errors='coerce').dt.year
            filtered_df = filtered_df[(filtered_df['year'] >= min_year) & (filtered_df['year'] <= max_year)]
            
            # Rating filter
            if 'vote_average' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]
            
            # Vote count filter
            if 'vote_count' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['vote_count'] >= min_votes]
            
            # Genre filter
            if selected_genres:
                def genre_filter(genres_str):
                    try:
                        import ast
                        genres = ast.literal_eval(genres_str) if isinstance(genres_str, str) else genres_str
                        return any(genre in genres for genre in selected_genres)
                    except:
                        return False
                
                filtered_df = filtered_df[filtered_df['genres'].apply(genre_filter)]
            
            # Sort by popularity or rating
            if 'popularity' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('popularity', ascending=False)
            elif 'vote_average' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('vote_average', ascending=False)
            
            st.success(f"🎬 Found {len(filtered_df)} movies matching your criteria!")
            
            # Display results
            for i, (_, movie) in enumerate(filtered_df.head(10).iterrows()):
                display_enhanced_movie_card(movie, rank=i+1)
                st.markdown("---")
                
        except Exception as e:
            st.error(f"Error filtering movies: {str(e)}")

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
        <h2>ℹ️ About FilmFusion Pro</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **🧠 Advanced AI Features:**
    
    🎭 **Enhanced Data Analysis**
    - Sentiment analysis integration
    - Cast & crew information
    - Multi-language support
    - Release date analytics
    
    🤖 **Machine Learning Algorithms**
    - Neural Collaborative Filtering
    - Multi-Armed Bandit RL
    - Content-based filtering
    - Hybrid recommendation systems
    
    📊 **Rich Visualizations**
    - Interactive charts & graphs
    - Sentiment distribution analysis
    - Language & year trends
    - User activity patterns
    
    🎬 **Movie Explorer**
    - Advanced filtering options
    - Genre-based discovery
    - Popularity & rating filters
    - Interactive movie browser
    """)
    
    st.markdown("---")
    
    # System Status with colors
    st.subheader("🔧 System Status")
    status_items = []
    
    if recommender:
        status_items.append("✅ Core System")
        if hasattr(recommender, 'user_profiles') and recommender.user_profiles:
            status_items.append("✅ User Profiles")
        if hasattr(recommender, 'movie_profiles') and recommender.movie_profiles:
            status_items.append("✅ Movie Profiles")
        if hasattr(recommender, 'global_bandit') and recommender.global_bandit:
            status_items.append("✅ Reinforcement Learning")
        if hasattr(recommender, 'evaluation_results') and recommender.evaluation_results:
            status_items.append("✅ Model Evaluation")
    
    for status in status_items:
        st.markdown(status)
    
    st.markdown("---")
    st.markdown("**🎬 Discover your next favorite movie with AI!**")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h3>🎬 FilmFusion Pro</h3>
    <p>Advanced Movie Recommendation System | Powered by Machine Learning & AI</p>
    <p>Enhanced with Sentiment Analysis & Rich Movie Data</p>
</div>
""", unsafe_allow_html=True)
