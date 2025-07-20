import streamlit as st
import pandas as pd
import os
import sys
import copy

# =============================================================================
# 1. SETUP: PAGE CONFIGURATION AND PATHS
# =============================================================================
st.set_page_config(page_title="MoviePulse", layout="wide")

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)
    from src.recommenders.recommender_system import RecommenderSystem
except ImportError as e:
    st.error(f"FATAL ERROR: Could not import project files: {e}")
    st.stop()

# =============================================================================
# 2. MODEL LOADING & DATA PREPARATION
# =============================================================================
@st.cache_resource
def load_main_recommender():
    """Loads the main, global recommender system."""
    loading_message = st.empty()
    loading_message.info("Initializing Recommender System... (This is a one-time setup and may take a minute.)")
    
    metadata_file = os.path.join(project_root, "data", "processed", "movies_metadata_enriched.csv")
    if not os.path.exists(metadata_file):
        st.error(f"FATAL ERROR: The required metadata file was not found at '{metadata_file}'.")
        st.error("Please run 'analysis/run_advanced_sentiment.py' to generate it.")
        st.stop()

    content_cols = ['overview', 'tagline', 'genres', 'cast', 'crew', 'keywords', 'reviews']
    recommender = RecommenderSystem(
        ratings_path=os.path.join(project_root, "data", "processed", "ratings_cleans.csv"),
        metadata_path=metadata_file,
        content_cols=content_cols,
        verbose=False 
    )
    recommender.run_all()
 
    
    
    movie_titles = sorted(recommender.metadata_df['title'].dropna().unique())
    loading_message.success("✅ Recommender System is ready!")
    return recommender, movie_titles

try:
    recommender, movie_titles = load_main_recommender()
except Exception as e:
    st.error("Failed to initialize the Recommender System.")
    st.exception(e)
    st.stop()

# =============================================================================
# 3. HELPER FUNCTION FOR DISPLAYING MOVIE DETAILS
# =============================================================================
def display_movie_details(row, score_type='Predicted Score'):
    """A helper function to neatly display movie details."""
    BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w200"
    poster_path = row.get('poster_path')
    poster_url = f"{BASE_IMAGE_URL}{poster_path}" if pd.notna(poster_path) else "https://placehold.co/200x300/000000/FFFFFF?text=No+Poster"
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(poster_url, width=150)
    with col2:
        st.markdown(f"#### {row['title']}")
        if 'tagline' in row and pd.notna(row['tagline']) and row['tagline']:
            st.markdown(f"*{row['tagline']}*")
        
        score_line = []
        if score_type == 'Predicted Score' and 'hybrid_score' in row:
            score_line.append(f"**Predicted Score:** `{row['hybrid_score']:.2f}`")
        elif score_type == 'Similarity Score' and 'similarity' in row:
            score_line.append(f"**Similarity Score:** `{row['similarity']:.3f}`")
        
        if 'vote_average' in row and row['vote_average'] > 0:
            score_line.append(f"**TMDB Rating:** `{row['vote_average']:.1f}/10`")
        
        if 'compound' in row:
             score_line.append(f"**Sentiment Score:** `{row['compound']:.2f}`")
        if score_line:
            st.markdown(" | ".join(score_line))

        details_line = []
        if 'genres' in row and pd.notna(row['genres']):
            details_line.append(f"**Genres:** `{row['genres']}`")
        if 'release_date' in row and pd.notna(row['release_date']):
            details_line.append(f"**Release Date:** `{row['release_date']}`")
        if 'runtime' in row and row['runtime'] > 0:
            details_line.append(f"**Runtime:** `{int(row['runtime'])} min`")
        if details_line:
            st.markdown("<br>".join(details_line), unsafe_allow_html=True)

        if 'overview' in row and pd.notna(row['overview']) and row['overview']:
            with st.expander("Overview"):
                st.write(row['overview'])
    st.markdown("---")

# =============================================================================
# 4. USER INTERFACE (UI)
# =============================================================================
st.title("🎬 MoviePulse Recommendation & Insights Engine")

tab1, tab2, tab3, tab4 = st.tabs(["👤 For You", "🎞️ Find Similar", "🚀 Cold Start", "📊 Sentiment & Trends"])

with tab1:
    st.header("Get Personalized Movie Recommendations")
    user_id_input = st.text_input("Enter an existing User ID:", placeholder="e.g., 1, 50, 250")
    if st.button("Get Your Recommendations", type="primary", key="user_rec_button"):
        if user_id_input:
            try:
                user_id = int(user_id_input)
                with st.spinner(f"Fetching recommendations for User {user_id}..."):
                    recommendations = recommender.recommend(user_id=user_id, strategy='hybrid', top_k=10)
                if not recommendations.empty:
                    st.subheader(f"Top 10 Recommendations for User {user_id}")
                    for _, row in recommendations.iterrows():
                        display_movie_details(row, score_type='Predicted Score')
                else:
                    st.warning(f"Could not generate recommendations for User {user_id}.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

with tab2:
    st.header("Find Movies Similar to a Title You Like")
    selected_movie = st.selectbox("Choose a movie:", options=movie_titles, index=None, placeholder="Search for a movie...")
    if selected_movie:
        st.markdown("---")
        st.subheader(f"You Selected: {selected_movie}")
        try:
            movie_details = recommender.metadata_df[recommender.metadata_df['title'] == selected_movie].iloc[0]
            display_movie_details(movie_details, score_type=None)
        except (IndexError, KeyError):
            st.error("Could not retrieve details for the selected movie.")
    if st.button("Find Similar Movies", type="primary", key="content_rec_button"):
        if selected_movie:
            try:
                with st.spinner(f"Finding movies similar to '{selected_movie}'..."):
                    recommendations = recommender.recommend_content_based(movie_title=selected_movie)
                if not recommendations.empty:
                    st.subheader(f"Top 10 Movies Similar to '{selected_movie}'")
                    for _, row in recommendations.iterrows():
                        display_movie_details(row, score_type='Similarity Score')
                else:
                    st.warning(f"Could not find similar movies for '{selected_movie}'.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.info("Please select a movie from the list first.")

with tab3:
    st.header("Get Recommendations for a New User (Cold Start)")
    st.info("Simulate a new user by uploading a CSV file of their movie ratings. The file must have two columns: `tmdbId` and `rating`.")
    uploaded_file = st.file_uploader("Upload your ratings CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            if not {'tmdbId', 'rating'}.issubset(user_df.columns):
                st.error("❌ Invalid format. CSV must have 'tmdbId' and 'rating' columns.")
            else:
                st.write("Your ratings:")
                st.dataframe(user_df)
                if st.button("🎯 Get Recommendations for This Profile", type="primary"):
                    with st.spinner("Creating temporary profile..."):
                        temp_recommender = copy.deepcopy(recommender)
                        synthetic_user_id = -99
                        liked_movie_ids = user_df[user_df['rating'] > 3.5]['tmdbId'].tolist()
                        if not liked_movie_ids:
                            st.warning("No movies with a rating > 3.5 found in your file.")
                        else:
                            temp_recommender.set_user_profile(user_id=synthetic_user_id, train_item_ids=liked_movie_ids)
                            result = temp_recommender.recommend(user_id=synthetic_user_id, strategy='hybrid', top_k=10)
                            if not result.empty:
                                st.subheader("Top 10 Recommendations Based on Your Uploaded Profile")
                                for _, row in result.iterrows():
                                    display_movie_details(row, score_type='Predicted Score')
                            else:
                                st.warning("Could not generate recommendations from this profile.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")


with tab4:
    st.header("📊 Audience Sentiment Insights")
    
    required_sentiment_cols = ['compound', 'num_reviews']
    if all(col in recommender.metadata_df.columns for col in required_sentiment_cols):
        st.markdown("Explore which movies are most loved or polarizing based on our **Smoothed Sentiment Score**.")

        sentiment_data = recommender.metadata_df[recommender.metadata_df['num_reviews'] > 0].copy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 Top 15 Most-Loved Movies")
            top_positive = sentiment_data.sort_values('compound', ascending=False).head(15)
            st.dataframe(top_positive[['title', 'compound', 'num_reviews']],
                         column_config={
                             "title": "Movie Title",
                             "compound": st.column_config.ProgressColumn(
                                 "Smoothed Sentiment", format="%.3f", min_value=0, max_value=1,
                             ),
                             "num_reviews": "Review Count"
                         }, use_container_width=True)

        with col2:
            st.subheader("📉 Top 15 Most-Polarizing Movies")
            top_negative = sentiment_data.sort_values('compound', ascending=True).head(15)
            st.dataframe(top_negative[['title', 'compound', 'num_reviews']],
                         column_config={
                             "title": "Movie Title",
                             "compound": st.column_config.ProgressColumn(
                                 "Smoothed Sentiment", format="%.3f", min_value=-1, max_value=0,
                             ),
                             "num_reviews": "Review Count"
                         }, use_container_width=True)
    
        st.markdown("---")
        recommendations = recommender.recommend(user_id=1, strategy='hybrid', top_k=10)
        recommendations[['title', 'compound', 'hybrid_score']]
        st.subheader("Smoothed Sentiment vs. TMDB Rating")
        st.scatter_chart(sentiment_data, x='vote_average', y='compound', color='#ff6347', size='num_reviews')
        st.caption("This chart shows the relationship between TMDB rating and our calculated sentiment score. Bubble size represents the number of reviews.")

    else:
        st.warning("Aggregated sentiment data not found!")
        st.info(
            "The required columns (e.g., 'compound') are missing. "
            "Please run the advanced sentiment analysis script:\n\n"
            "`python analysis/run_advanced_sentiment.py`"
        )

st.markdown("---")
st.markdown("Built with ❤️ by the FilmFusion Team.")
