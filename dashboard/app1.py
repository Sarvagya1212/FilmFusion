import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
import time
import os
from typing import Optional
import sys

# Import your RecommenderSystem class
# from recommender_system import RecommenderSystem

st.set_page_config(
    page_title="FilmFusion - Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    from src.recommenders.recommender_system import RecommenderSystem
except ImportError as e:
    st.error(f"FATAL ERROR: Could not import project files: {e}")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    .movie-poster {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        max-width: 100%;
        height: auto;
    }
    .movie-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .movie-overview {
        color: #5a6c7d;
        line-height: 1.5;
        margin-bottom: 1rem;
        text-align: justify;
    }
    .movie-details {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    .detail-badge {
        background: #e9ecef;
        color: #495057;
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
    .genre-badge {
        background: #d1ecf1;
        color: #0c5460;
        padding: 0.3rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: 500;
    }
    .rating-score {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
    }
    .user-mode {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .admin-mode {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .cast-list {
        color: #6c757d;
        font-size: 0.9em;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'user'
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {'user_id': 1, 'preferences': []}

def get_poster_url(poster_path, size='w500'):
    """Generate full poster URL from TMDB poster path"""
    if pd.isna(poster_path) or poster_path is None or str(poster_path) == 'nan':
        return "https://via.placeholder.com/500x750/cccccc/666666?text=No+Poster"
    
    base_url = "https://image.tmdb.org/t/p/"
    if not str(poster_path).startswith('/'):
        poster_path = '/' + str(poster_path)
    
    return f"{base_url}{size}{poster_path}"

def format_runtime(runtime):
    """Format runtime in minutes to hours and minutes"""
    if pd.isna(runtime) or runtime == 0:
        return "Unknown"
    
    try:
        runtime = int(float(runtime))
        hours = runtime // 60
        minutes = runtime % 60
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    except:
        return "Unknown"

def format_cast(cast_str, max_actors=5):
    """Format cast string to show limited number of actors"""
    if pd.isna(cast_str) or cast_str == '' or str(cast_str) == 'nan':
        return "Cast information not available"
    
    try:
        # Handle different cast formats
        if isinstance(cast_str, str):
            # If it's a comma-separated string
            if ',' in cast_str:
                actors = [actor.strip() for actor in cast_str.split(',')]
            else:
                actors = [cast_str.strip()]
            
            # Limit number of actors shown
            if len(actors) > max_actors:
                return ', '.join(actors[:max_actors]) + f" and {len(actors) - max_actors} more"
            else:
                return ', '.join(actors)
        else:
            return str(cast_str)
    except:
        return "Cast information not available"

def format_genres(genres_str):
    """Format genres string into badges"""
    if pd.isna(genres_str) or genres_str == '' or str(genres_str) == 'nan':
        return ""
    
    try:
        # Handle different genre formats
        if isinstance(genres_str, str):
            if '|' in genres_str:
                genres = [genre.strip() for genre in genres_str.split('|')]
            elif ',' in genres_str:
                genres = [genre.strip() for genre in genres_str.split(',')]
            else:
                genres = [genres_str.strip()]
            
            # Create HTML badges for genres
            badges = []
            for genre in genres[:5]:  # Limit to 5 genres
                if genre:
                    badges.append(f'<span class="genre-badge">{genre}</span>')
            
            return ' '.join(badges)
        else:
            return f'<span class="genre-badge">{str(genres_str)}</span>'
    except:
        return ""

def load_default_system():
    """Load the recommender system with default/production data"""
    try:
        # In production, these would be your actual data files
        default_ratings = r"C:\Users\sarva\MoviePulse\data\processed\ratings_cleans.csv"  # Your production ratings
        default_metadata = r"C:\Users\sarva\MoviePulse\data\processed\movies_with_sentiment.csv"  # Your movie catalog
        
        if os.path.exists(default_ratings) and os.path.exists(default_metadata):
            with st.spinner('Loading movie database...'):
                recommender = RecommenderSystem(
                    ratings_path=default_ratings,
                    metadata_path=default_metadata,
                    verbose=False  # Less verbose for user experience
                )
                recommender.run_all()
                st.session_state.recommender = recommender
                st.session_state.data_loaded = True
                return True
        else:
            st.error("Movie database not found. Please contact administrator.")
            return False
    except Exception as e:
        st.error(f"Error loading movie database: {str(e)}")
        return False

def load_custom_system(ratings_file, metadata_file):
    """Load recommender system with custom data (admin/research mode)"""
    try:
        with st.spinner('Loading custom dataset...'):
            # Save uploaded files temporarily if they're file uploads
            if hasattr(ratings_file, 'read'):
                ratings_path = f"temp_ratings_{int(time.time())}.csv"
                with open(ratings_path, 'wb') as f:
                    f.write(ratings_file.read())
            else:
                ratings_path = ratings_file
                
            if hasattr(metadata_file, 'read'):
                metadata_path = f"temp_metadata_{int(time.time())}.csv"
                with open(metadata_path, 'wb') as f:
                    f.write(metadata_file.read())
            else:
                metadata_path = metadata_file
            
            recommender = RecommenderSystem(
                ratings_path=ratings_path,
                metadata_path=metadata_path,
                verbose=True
            )
            recommender.run_all()
            st.session_state.recommender = recommender
            st.session_state.data_loaded = True
            
            # Clean up temp files
            if hasattr(ratings_file, 'read') and os.path.exists(ratings_path):
                os.remove(ratings_path)
            if hasattr(metadata_file, 'read') and os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            return True
    except Exception as e:
        st.error(f"Error loading custom dataset: {str(e)}")
        return False

def display_movie_search():
    """Movie search interface for content-based recommendations"""
    st.subheader("🔍 Find Similar Movies")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        # Get all movie titles from the recommender system
        if st.session_state.data_loaded and st.session_state.recommender:
            # Fix: Filter out non-string values and handle NaN/None values
            titles_series = st.session_state.recommender.metadata_df['title']
            
            # Filter to only include valid string titles
            all_movies = [
                str(title).strip() 
                for title in titles_series.unique() 
                if pd.notna(title) and 
                   str(title).strip() != '' and 
                   str(title).strip().lower() not in ['nan', 'none', 'null']
            ]
            
            # Remove any remaining invalid entries
            all_movies = [movie for movie in all_movies if movie and len(movie) > 0]
            
            # Sort alphabetically (now safe since all are strings)
            all_movies.sort()
            
            selected_movie = st.selectbox(
                "Choose a movie:", 
                options=[""] + all_movies,  # Empty option first
                help="Select a movie to find similar recommendations"
            )
        else:
            selected_movie = st.text_input(
                "Search for a movie:", 
                placeholder="System not loaded...",
                disabled=True
            )
    
    with col2:
        num_suggestions = st.slider("Number of suggestions:", 5, 15, 8)
    
    if selected_movie and selected_movie != "":
        if st.button("🎬 Find Similar Movies", type="primary"):
            try:
                recommendations = st.session_state.recommender.recommend(
                    movie_title=selected_movie,
                    strategy='content',
                    top_k=num_suggestions
                )
                display_user_recommendations(recommendations, f"Movies similar to '{selected_movie}'")
            except Exception as e:
                st.error(f"Error finding similar movies: {str(e)}")


def display_personalized_recommendations():
    """Personalized recommendations for logged-in users"""
    st.subheader("🎯 Your Personalized Recommendations")
    
    user_id = st.session_state.user_profile['user_id']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        algorithm = st.selectbox(
            "Recommendation Style:",
            ["Smart Blend", "Similar Users", "Similar Movies", "Advanced AI"],
            help="Different ways to find movies you'll love"
        )
    with col2:
        num_recs = st.slider("Number of recommendations:", 5, 20, 10)
    with col3:
        show_seen = st.checkbox("Include movies I've seen", False)
    
    # Map user-friendly names to actual strategies
    strategy_map = {
        "Smart Blend": "hybrid",
        "Similar Users": "user", 
        "Similar Movies": "item",
        "Advanced AI": "svd"
    }
    
    if st.button("✨ Get My Recommendations", type="primary"):
        try:
            recommendations = st.session_state.recommender.recommend(
                user_id=user_id,
                strategy=strategy_map[algorithm],
                top_k=num_recs,
                filter_seen=not show_seen
            )
            display_user_recommendations(recommendations, f"Personalized for you ({algorithm})")
        except Exception as e:
            st.error(f"Unable to generate recommendations: {str(e)}")
def display_user_recommendations(recommendations_df, title):
    """Display recommendations in a user-friendly format with enhanced movie cards"""
    if recommendations_df.empty:
        st.warning("No recommendations found. Try adjusting your preferences!")
        return
    
    st.markdown(f"### {title}")
    
    # Display movies in a single column for better readability
    for i, row in recommendations_df.head(10).iterrows():
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Movie poster
                poster_url = get_poster_url(row.get('poster_path'))
                st.markdown(f'<img src="{poster_url}" class="movie-poster" style="width: 100%; max-width: 200px;">', unsafe_allow_html=True)
            
            with col2:
                # Movie title
                movie_title = row.get('title', 'Unknown Title')
                st.markdown(f'<div class="movie-title">🎬 {movie_title}</div>', unsafe_allow_html=True)
                
                # Rating and score
                col2a, col2b, col2c = st.columns([2, 2, 2])
                
                with col2a:
                    vote_avg = row.get('vote_average', 'N/A')
                    if vote_avg != 'N/A':
                        try:
                            vote_avg = f"{float(vote_avg):.1f}/10"
                        except:
                            vote_avg = 'N/A'
                    st.markdown(f"⭐ **TMDB:** {vote_avg}")
                
                with col2b:
                    release_date = str(row.get('release_date', 'Unknown'))[:4]
                    st.markdown(f"📅 **Year:** {release_date}")
                
                with col2c:
                    # Recommendation score
                    score = ""
                    if 'predicted_rating' in row:
                        score = f"⭐ {row['predicted_rating']:.1f}"
                    elif 'similarity' in row:
                        score = f"🎯 {row['similarity']:.2f}"
                    elif 'hybrid_score' in row:
                        score = f"✨ {row['hybrid_score']:.2f}"
                    
                    if score:
                        st.markdown(f'<span class="rating-score">{score}</span>', unsafe_allow_html=True)
                
                # Movie overview
                overview = row.get('overview', 'No description available')
                if overview and overview != 'No description available':
                    # Truncate overview if too long
                    if len(overview) > 300:
                        overview = overview[:300] + "..."
                    st.markdown(f'<div class="movie-overview">{overview}</div>', unsafe_allow_html=True)
                
                # Additional details row
                col3a, col3b = st.columns([1, 1])
                
                with col3a:
                    # Runtime and language
                    runtime = format_runtime(row.get('runtime'))
                    language = row.get('language', 'Unknown')
                    if language != 'Unknown':
                        language = language.upper()
                    
                    st.markdown(f'<div class="movie-details">'
                              f'<span class="detail-badge">⏱️ {runtime}</span>'
                              f'<span class="detail-badge">🗣️ {language}</span>'
                              f'</div>', unsafe_allow_html=True)
                
                with col3b:
                    # Genres
                    genres_html = format_genres(row.get('genres'))
                    if genres_html:
                        st.markdown(f'<div class="movie-details">{genres_html}</div>', unsafe_allow_html=True)
                
                # Cast information
                cast_info = format_cast(row.get('cast', ''))
                if cast_info != "Cast information not available":
                    st.markdown(f'<div class="cast-list">👥 **Cast:** {cast_info}</div>', unsafe_allow_html=True)
                
                # Separator line
                st.markdown("<hr style='margin: 1.5rem 0; border: none; height: 1px; background: #e9ecef;'>", unsafe_allow_html=True)

def display_admin_interface():
    """Admin interface for system management and testing"""
    st.markdown('<div class="admin-mode"><h3>🔧 Admin Panel</h3></div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 System Status", "🔄 Data Management", "🧪 Algorithm Testing", "📈 Analytics"])
    
    with tab1:
        st.subheader("System Status")
        if st.session_state.data_loaded and st.session_state.recommender:
            col1, col2, col3 = st.columns(3)
            with col1:
                num_users = len(st.session_state.recommender.user_item_matrix.index)
                st.metric("Active Users", f"{num_users:,}")
            with col2:
                num_movies = len(st.session_state.recommender.metadata_df)
                st.metric("Movies in Catalog", f"{num_movies:,}")
            with col3:
                num_ratings = len(st.session_state.recommender.ratings_df)
                st.metric("Total Ratings", f"{num_ratings:,}")
        else:
            st.warning("System not loaded")
    
    with tab2:
        st.subheader("Data Management")
        
        data_source = st.radio("Data Source:", ["Production Database", "Custom Dataset"])
        
        if data_source == "Production Database":
            if st.button("🔄 Reload Production Data"):
                if load_default_system():
                    st.success("Production data reloaded successfully!")
                    st.rerun()
        else:
            st.write("Upload custom dataset for testing:")
            ratings_file = st.file_uploader("Ratings CSV", type=['csv'])
            metadata_file = st.file_uploader("Movies Metadata CSV", type=['csv'])
            
            if ratings_file and metadata_file:
                if st.button("📤 Load Custom Dataset"):
                    if load_custom_system(ratings_file, metadata_file):
                        st.success("Custom dataset loaded successfully!")
                        st.rerun()
    
    with tab3:
        st.subheader("Algorithm Testing")
        if st.session_state.data_loaded:
            test_user = st.number_input("Test User ID:", min_value=1, value=1)
            
            strategies = ["hybrid", "user", "item", "svd"]
            results = {}
            
            if st.button("🧪 Run A/B Test"):
                with st.spinner("Testing all algorithms..."):
                    for strategy in strategies:
                        try:
                            recs = st.session_state.recommender.recommend(
                                user_id=test_user,
                                strategy=strategy,
                                top_k=5
                            )
                            results[strategy] = len(recs)
                        except:
                            results[strategy] = 0
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                cols = [col1, col2, col3, col4]
                for i, (strategy, count) in enumerate(results.items()):
                    with cols[i]:
                        st.metric(f"{strategy.title()}", f"{count} recs")
        else:
            st.warning("Load data first")
    
    with tab4:
        st.subheader("System Analytics")
        if st.session_state.recommendations is not None:
            create_admin_visualizations(st.session_state.recommendations)
        else:
            st.info("Generate recommendations to see analytics")

def create_admin_visualizations(recommendations_df):
    """Create detailed analytics for admins"""
    col1, col2 = st.columns(2)
    
    with col1:
        if 'vote_average' in recommendations_df.columns:
            fig = px.histogram(recommendations_df, x='vote_average', 
                             title="TMDB Rating Distribution",
                             color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'release_date' in recommendations_df.columns:
            recommendations_df['release_year'] = pd.to_datetime(
                recommendations_df['release_date'], errors='coerce'
            ).dt.year
            
            year_counts = recommendations_df['release_year'].value_counts().sort_index()
            fig = px.line(x=year_counts.index, y=year_counts.values,
                         title="Release Year Trends",
                         color_discrete_sequence=['#4ECDC4'])
            fig.update_layout(xaxis_title="Year", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🎬 FilmFusion</h1>', 
               unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">Your Personal Movie Discovery Platform</p>', 
               unsafe_allow_html=True)
    
    # Mode selection in sidebar
    with st.sidebar:
        st.header("🎭 Mode Selection")
        app_mode = st.radio(
            "Choose your role:",
            ["🎬 Movie Fan", "🔧 Admin/Research"],
            help="Movie Fan: Find great movies to watch\nAdmin/Research: Test algorithms and manage data"
        )
        
        st.session_state.app_mode = 'admin' if 'Admin' in app_mode else 'user'
        
        # User profile section
        if st.session_state.app_mode == 'user':
            st.header("👤 Your Profile")
            user_id = st.number_input("User ID:", min_value=1, value=st.session_state.user_profile['user_id'])
            st.session_state.user_profile['user_id'] = user_id
            
            # Quick rating interface could go here
            st.info("💡 Rate movies to get better recommendations!")
        
        # System initialization
        if not st.session_state.data_loaded:
            st.header("⚙️ System Status")
            st.warning("🔄 Loading movie database...")
            if st.button("🚀 Initialize System"):
                load_default_system()
                st.rerun()
        else:
            st.success("✅ System Ready")
    
    # Auto-load system on first run
    if not st.session_state.data_loaded and st.session_state.app_mode == 'user':
        load_default_system()
    
    # Main content based on mode
    if st.session_state.app_mode == 'user':
        display_user_mode()
    else:
        display_admin_interface()

def display_user_mode():
    """Main user interface for movie recommendations"""
    if not st.session_state.data_loaded:
        st.markdown('<div class="user-mode"><h3>🎬 Welcome to Filmfusion!</h3><p>We\'re preparing your personalized movie recommendations...</p></div>', 
                   unsafe_allow_html=True)
        return
    
    st.markdown('<div class="user-mode"><h3>🎬 Discover Your Next Favorite Movie</h3></div>', 
               unsafe_allow_html=True)
    
    # Two main recommendation types
    tab1, tab2 = st.tabs(["🎯 For You", "🔍 Find Similar"])
    
    with tab1:
        display_personalized_recommendations()
    
    with tab2:
        display_movie_search()

if __name__ == "__main__":
    main()