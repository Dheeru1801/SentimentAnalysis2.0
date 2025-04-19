"""
Data utilities for loading and managing datasets.
"""

import streamlit as st
import pandas as pd
import os

# Change to absolute import
from config import settings

@st.cache_data
def load_data():
    """
    Load the dataset for sentiment analysis.
    
    Returns:
        DataFrame: Pandas DataFrame containing the loaded data
    """
    try:
        # Try to load the dataset
        data = pd.read_csv(settings.get_data_path('Dataset-SA.csv')).sample(frac=0.1, random_state=42)
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        # Return a sample dataset if the actual data can't be loaded
        sample_data = pd.DataFrame({
            'Summary': ['This product is amazing!', 'I hate this product.', 'It was okay.'],
            'Sentiment': ['Positive', 'Negative', 'Neutral']
        })
        return sample_data