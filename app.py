import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

from scraper import SocialMediaScraper
from preprocessor import DataPreprocessor
from ml_model import StockPredictor

def load_credentials():
    """
    Load Reddit API credentials from environment variables.
    
    Returns:
        dict: Credentials dictionary
    """
    load_dotenv()
    return {
        'client_id': os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'user_agent': os.getenv('REDDIT_USER_AGENT')
    }

def local_css(file_name):
    """
    Load a local CSS file for custom styling.
    
    Args:
        file_name (str): Path to the CSS file
    """
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    
    # Page configuration
    st.set_page_config(
        page_title="Reddit Sentiment Predictor", 
        page_icon=":chart_with_upwards_trend:", 
        layout="wide"
    )
    
    # Apply custom CSS
    local_css("styles.css")
    # Custom header with added styling class
    st.markdown("""
    <div class="app-header">
        <h1>🚀 Reddit Sentiment Stock Predictor</h1>
        <p class="subtitle">Analyze stock sentiment through Reddit discussions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.header('🔍 Analysis Parameters')
    subreddit = st.sidebar.text_input('Subreddit', 'stocks', help="Choose the subreddit to analyze")
    post_limit = st.sidebar.slider('Post Limit', 50, 1000, 100, help="Number of posts to scrape")
    sort_method = st.sidebar.selectbox(
        'Sort Posts By', 
        ['New', 'Hot', 'Top'], 
        index=0,
        help="Method to sort Reddit posts"
    )

    # Load credentials
    try:
        credentials = load_credentials()
    except Exception as e:
        st.error(f"Credential loading error: {e}")
        return

    # Main workflow
    if st.button('Analyze Subreddit', help="Click to start Reddit sentiment analysis"):
        with st.spinner('Scraping and Analyzing Data...'):
            try:
                # Initialize and scrape
                scraper = SocialMediaScraper(
                    credentials['client_id'], 
                    credentials['client_secret'], 
                    credentials['user_agent'], 
                    subreddit=subreddit
                )
                raw_data = scraper.scrape(
                    limit=post_limit, 
                    sort_by=sort_method.lower()
                )

                # Preprocess data
                preprocessor = DataPreprocessor(raw_data)
                preprocessor.clean_text().extract_sentiment_features()
                preprocessor.clean_text().normalize_features(columns=['score','comments'])
                preprocessed_data = preprocessor.get_processed_data()

                # Predict
                predictor = StockPredictor(preprocessed_data)
                prediction_results = predictor.train_model()

                # Visualizations with custom styling
                st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Total Posts</h3>
                        <div class="metric-value">{len(preprocessed_data)}</div>
                        <p class="metric-subtitle">Subreddit Analysis</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Model R² Score</h3>
                        <div class="metric-value">{prediction_results['r2']:.2f}</div>
                        <p class="metric-subtitle">Prediction Accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Mean Absolute Error</h3>
                        <div class="metric-value">{prediction_results['mae']:.4f}</div>
                        <p class="metric-subtitle">Prediction Error</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Sentiment Distribution
                st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
                sentiment_cols = [col for col in preprocessed_data.columns if col.startswith('sentiment_feature_')]
                sentiment_data = preprocessed_data[sentiment_cols].mean()
                
                if not sentiment_data.empty and sentiment_data.sum() > 0:
                    sentiment_df = sentiment_data.reset_index()
                    sentiment_df.columns = ['Feature', 'Average Importance']
                    fig_sentiment = px.bar(
                        sentiment_df,
                        x='Feature', 
                        y='Average Importance',
                        labels={'x': 'Sentiment Features', 'y': 'Average Importance'},
                        title='Sentiment Feature Importance'
                    )
                    st.plotly_chart(fig_sentiment)
                else:
                    st.warning("No sentiment features found in data")

                # Post Score Distribution
                st.markdown('<div class="section-header">Post Score Distribution</div>', unsafe_allow_html=True)
                fig_scores = px.histogram(
                    preprocessed_data, 
                    x='normalized_score', 
                    title='Distribution of Normalized Post Scores'
                )
                st.plotly_chart(fig_scores)

                # Model Performance Details
                st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
                st.write("Cross-Validation Scores:", prediction_results['cross_val_scores'])
                
                # Raw Data Preview
                st.markdown('<div class="section-header">Raw Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(preprocessed_data.head())

            except Exception as e:
                st.error(f"Analysis error: {e}")

if __name__ == '__main__':
    main()