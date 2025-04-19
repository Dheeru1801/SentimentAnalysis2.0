"""
Configuration settings for the Sentiment Analysis Dashboard.
"""

# Page Configuration
PAGE_TITLE = "Sentiment Analysis Dashboard"
PAGE_ICON = "😊"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Sample text for demonstration
SAMPLE_TEXT = "This product is amazing! I love how it works. The quality is top-notch."

# Model settings
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_RANDOM_STATE = 42

# Sentiment thresholds
POSITIVE_THRESHOLD = 0.4
NEGATIVE_THRESHOLD = 0.4

# WordCloud settings
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_MAX_WORDS = 200
WORDCLOUD_MAX_FONT_SIZE = 40

# Color schemes for visualizations
COLORS = {
    'Positive': '#10B981',
    'Negative': '#EF4444',
    'Neutral': '#6B7280'
}

# File paths helper
import os

def get_data_path(filename):
    """Helper function for file paths"""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), filename)