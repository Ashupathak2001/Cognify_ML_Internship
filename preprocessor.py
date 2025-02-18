import pandas as pd
import numpy as np
import re
from typing import Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize preprocessor with input DataFrame and spaCy NLP model.
        
        Args:
            df (pd.DataFrame): Input DataFrame to preprocess
        """
        self.df = df.copy()
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy model

    def clean_text(self, column: str = 'cleaned_text') -> 'DataPreprocessor':
        """
        Clean and preprocess text data using spaCy.
        
        Args:
            column (str, optional): Column name for cleaned text. Defaults to 'cleaned_text'.
        
        Returns:
            DataPreprocessor: Instance with cleaned text
        """
        # Ensure required columns exist
        self.df['title'] = self.df.get('title', '')
        self.df['selftext'] = self.df.get('selftext', '')
        
        # Combine text sources
        self.df[column] = self.df['title'] + " " + self.df['selftext']
        
        # Advanced text cleaning using spaCy
        self.df[column] = self.df[column].apply(self._advanced_clean)
        
        return self

    def _advanced_clean(self, text: str) -> str:
        """
        Advanced text cleaning method using spaCy.
        
        Args:
            text (str): Input text to clean
        
        Returns:
            str: Cleaned and tokenized text
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Keep lemmatized tokens that are not stopwords or punctuation
        tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        return ' '.join(tokens)

    def extract_sentiment_features(self, column: str = 'cleaned_text') -> 'DataPreprocessor':
        """
        Extract sentiment-related features using TF-IDF.
        
        Args:
            column (str, optional): Text column to analyze. Defaults to 'cleaned_text'.
        
        Returns:
            DataPreprocessor: Instance with sentiment features
        """

        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        
        vectorizer = TfidfVectorizer(max_features=100)
        sentiment_matrix = vectorizer.fit_transform(self.df[column].fillna(''))
        
        # Add sentiment features
        sentiment_df = pd.DataFrame(
            sentiment_matrix.toarray(), 
            columns=[f'sentiment_feature_{i}' for i in range(sentiment_matrix.shape[1])]
        )
        
        self.df = pd.concat([self.df, sentiment_df], axis=1)
        
        return self

    def normalize_features(self, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Normalize specified numerical columns.
        
        Args:
            columns (list, optional): Columns to normalize. Defaults to score and comments.
        
        Returns:
            pd.DataFrame: DataFrame with normalized features
        """
        if columns is None:
            columns = ['score', 'comments']
        
        for col in columns:
            if col in self.df.columns:
                # Use StandardScaler for better handling of outliers
                scaler = StandardScaler()
                self.df[f'normalized_{col}'] = scaler.fit_transform(
                    self.df[col].values.reshape(-1, 1)
                )
        
        return self.df

    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the processed DataFrame.
        
        Returns:
            pd.DataFrame: Processed data
        """
        return self.df
