import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import StringIO
import time
import os
import sys

# Import your RecommenderSystem class
# from recommender_system import RecommenderSystem

st.set_page_config(
    page_title="Movie Recommender System",
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
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
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

def load_recommender_system():
    """Initialize and load the recommender system"""
    try:
        # Initialize the recommender system
        # You'll need to provide the actual file paths
        ratings_path = st.session_state.get('ratings_file', 'ratings.csv')
        metadata_path = st.session_state.get('metadata_file', 'movies_metadata.csv')
        
        if ratings_path and metadata_path:
            with st.spinner('Initializing recommender system...'):
                recommender = RecommenderSystem(
                    ratings_path=ratings_path,
                    metadata_path=metadata_path,
                    verbose=True
                )
                recommender.run_all()
                st.session_state.recommender = recommender
                st.session_state.data_loaded = True
                st.success("✅ Recommender system loaded successfully!")
                return True
    except Exception as e:
        st.error(f"Error loading recommender system: {str(e)}")
        return False
    return False

def display_recommendations(recommendations_df, strategy):
    """Display recommendations in an attractive format"""
    if recommendations_df.empty:
        st.warning("No recommendations found!")
        return
    
    st.subheader(f"🎯 Top Recommendations ({strategy.title()})")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(recommendations_df)}</h3><p>Movies Found</p></div>', 
                   unsafe_allow_html=True)
    with col2:
        if 'predicted_rating' in recommendations_df.columns:
            avg_score = recommendations_df['predicted_rating'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_score:.2f}</h3><p>Avg Predicted Rating</p></div>', 
                       unsafe_allow_html=True)
        elif 'similarity' in recommendations_df.columns:
            avg_sim = recommendations_df['similarity'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_sim:.3f}</h3><p>Avg Similarity</p></div>', 
                       unsafe_allow_html=True)
        elif 'hybrid_score' in recommendations_df.columns:
            avg_score = recommendations_df['hybrid_score'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_score:.3f}</h3><p>Avg Hybrid Score</p></div>', 
                       unsafe_allow_html=True)
    with col3:
        if 'vote_average' in recommendations_df.columns:
            avg_rating = recommendations_df['vote_average'].mean()
            st.markdown(f'<div class="metric-card"><h3>{avg_rating:.1f}</h3><p>Avg TMDB Rating</p></div>', 
                       unsafe_allow_html=True)
    
    # Display recommendations
    for idx, row in recommendations_df.head(10).iterrows():
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>🎬 {row.get('title', 'Unknown Title')}</h4>
                <p><strong>Overview:</strong> {row.get('overview', 'No overview available')[:200]}...</p>
                <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                    <span><strong>Release:</strong> {row.get('release_date', 'Unknown')}</span>
                    <span><strong>Rating:</strong> {row.get('vote_average', 'N/A')}/10</span>
                    <span><strong>Genre:</strong> {str(row.get('genres', 'Unknown'))[:30]}...</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

def create_visualizations(recommendations_df, strategy):
    """Create visualizations for the recommendations"""
    if recommendations_df.empty:
        return
    
    st.subheader("📊 Recommendation Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        if 'predicted_rating' in recommendations_df.columns:
            fig = px.histogram(recommendations_df, x='predicted_rating', 
                             title="Distribution of Predicted Ratings",
                             color_discrete_sequence=['#FF6B6B'])
            st.plotly_chart(fig, use_container_width=True)
        elif 'similarity' in recommendations_df.columns:
            fig = px.histogram(recommendations_df, x='similarity', 
                             title="Distribution of Similarity Scores",
                             color_discrete_sequence=['#4ECDC4'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Release year distribution
        if 'release_date' in recommendations_df.columns:
            # Extract year from release_date
            recommendations_df['release_year'] = pd.to_datetime(
                recommendations_df['release_date'], errors='coerce'
            ).dt.year
            
            year_counts = recommendations_df['release_year'].value_counts().sort_index()
            fig = px.bar(x=year_counts.index, y=year_counts.values,
                        title="Movies by Release Year",
                        color_discrete_sequence=['#95E1D3'])
            fig.update_layout(xaxis_title="Year", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# Main App
def main():
    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', 
               unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload or path input
        st.subheader("📁 Data Files")
        upload_option = st.radio("Choose data input method:", 
                                ["Upload Files", "Use File Paths"])
        
        if upload_option == "Upload Files":
            ratings_file = st.file_uploader("Upload Ratings CSV", type=['csv'])
            metadata_file = st.file_uploader("Upload Movies Metadata CSV", type=['csv'])
            
            if ratings_file and metadata_file:
                # Save uploaded files temporarily
                st.session_state.ratings_file = ratings_file
                st.session_state.metadata_file = metadata_file
        else:
            ratings_path = st.text_input("Ratings CSV Path", "ratings.csv")
            metadata_path = st.text_input("Metadata CSV Path", "movies_metadata.csv")
            st.session_state.ratings_file = ratings_path
            st.session_state.metadata_file = metadata_path
        
        # Load system button
        if st.button("🚀 Initialize System", type="primary"):
            if load_recommender_system():
                st.rerun()
        
        # System status
        if st.session_state.data_loaded:
            st.success("✅ System Ready")
        else:
            st.warning("⏳ System Not Loaded")
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👆 Please configure and initialize the system using the sidebar.")
        
        # Show example of what the app can do
        st.subheader("🌟 What this app can do:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **Content-Based**
            - Find movies similar to your favorites
            - Based on genres, cast, plot
            """)
        with col2:
            st.markdown("""
            **Collaborative Filtering**
            - User-based recommendations
            - Item-based suggestions
            """)
        with col3:
            st.markdown("""
            **Advanced Methods**
            - Matrix factorization (SVD)
            - Hybrid approaches
            """)
        
        return
    
    # Recommendation interface
    st.subheader("🎯 Get Recommendations")
    
    # Strategy selection
    strategy = st.selectbox(
        "Choose Recommendation Strategy:",
        ["hybrid", "content", "user", "item", "svd"],
        help="Select the type of recommendation algorithm to use"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if strategy == "content":
            # Movie title input for content-based
            movie_title = st.text_input("Enter a movie title:", 
                                       placeholder="e.g., The Matrix")
            user_id = None
        else:
            # User ID input for other strategies
            user_id = st.number_input("Enter User ID:", min_value=1, value=1)
            movie_title = None
    
    with col2:
        top_k = st.slider("Number of recommendations:", 5, 20, 10)
        filter_seen = st.checkbox("Filter already seen movies", value=True)
    
    # Advanced options for hybrid
    if strategy == "hybrid":
        with st.expander("🔧 Advanced Hybrid Settings"):
            col1, col2, col3 = st.columns(3)
            with col1:
                alpha = st.slider("User-based weight (α):", 0.0, 1.0, 0.8, 0.1)
            with col2:
                beta = st.slider("Item-based weight (β):", 0.0, 1.0, 0.2, 0.1)
            with col3:
                delta = st.slider("Sentiment weight (δ):", 0.0, 1.0, 0.0, 0.1)
    else:
        alpha, beta, delta = 0.8, 0.2, 0.0
    
    # Get recommendations button
    if st.button("🎬 Get Recommendations", type="primary"):
        try:
            with st.spinner('Generating recommendations...'):
                if strategy == "content" and movie_title:
                    recommendations = st.session_state.recommender.recommend(
                        movie_title=movie_title,
                        strategy=strategy,
                        top_k=top_k
                    )
                elif user_id and strategy != "content":
                    recommendations = st.session_state.recommender.recommend(
                        user_id=user_id,
                        strategy=strategy,
                        top_k=top_k,
                        filter_seen=filter_seen,
                        alpha=alpha,
                        beta=beta,
                        delta=delta
                    )
                else:
                    st.error("Please provide the required input for the selected strategy.")
                    return
                
                st.session_state.recommendations = recommendations
                st.session_state.current_strategy = strategy
                
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
    
    # Display results
    if st.session_state.recommendations is not None:
        recommendations = st.session_state.recommendations
        current_strategy = st.session_state.get('current_strategy', 'unknown')
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📊 Analytics", "📄 Raw Data"])
        
        with tab1:
            display_recommendations(recommendations, current_strategy)
        
        with tab2:
            create_visualizations(recommendations, current_strategy)
        
        with tab3:
            st.subheader("Raw Recommendation Data")
            st.dataframe(recommendations, use_container_width=True)
            
            # Download button
            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations CSV",
                data=csv,
                file_name=f'recommendations_{current_strategy}_{int(time.time())}.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()