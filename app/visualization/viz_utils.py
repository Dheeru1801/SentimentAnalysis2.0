"""
Visualization utilities for the Sentiment Analysis Dashboard.
"""

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import nltk
import streamlit as st
from nltk.corpus import stopwords

# Change to absolute import
from config import settings

def generate_wordcloud(text, title="Word Cloud", width=settings.WORDCLOUD_WIDTH, height=settings.WORDCLOUD_HEIGHT):
    """
    Generate a word cloud visualization for the given text.
    
    Args:
        text (str): Text to visualize
        title (str): Title for the visualization
        width (int): Width of the word cloud
        height (int): Height of the word cloud
        
    Returns:
        fig: Matplotlib figure object containing the visualization
    """
    try:
        # Try using a fallback font or system font
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=settings.WORDCLOUD_MAX_WORDS,
            max_font_size=settings.WORDCLOUD_MAX_FONT_SIZE,
            random_state=42,
            prefer_horizontal=0.9,
            # Set a common system font that is likely to be available
            font_path=None,  # Let WordCloud choose available system font
            relative_scaling=0.5,
            min_font_size=10,
            regexp=r"\w[\w']+",
            collocations=False  # Avoid duplicate phrases
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        return fig
    except Exception as e:
        # Fallback to a better word frequency chart if WordCloud fails
        
        # Process text to get word frequencies
        words = text.lower().split()
        # Remove very short words and common stop words
        stop_words = set(stopwords.words('english') if nltk.data.find('corpora/stopwords') else 
                      ['a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
                       'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                       'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                       'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
                       'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how'])
        
        words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        
        # Get top words (maximum 10 for readability)
        num_words = min(10, len(word_freq))
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:num_words])
        
        # Create a horizontal bar chart for better readability
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use different colors for bars to make chart more visually appealing
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_words)))
        
        # Plot horizontally for better word display
        y_pos = range(len(top_words))
        ax.barh(y_pos, list(top_words.values()), color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(top_words.keys()))
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_title(f"{title} - Word Frequencies", fontsize=16)
        ax.set_xlabel('Frequency')
        
        # Add frequency values at the end of each bar
        for i, v in enumerate(top_words.values()):
            ax.text(v + 0.1, i, str(v), va='center')
            
        plt.tight_layout()
        return fig

def load_css():
    css = """
    <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }

        h1, h2, h3 {
            color: #FFFFFF !important;
        }

        .dashboard-card {
            background-color: #034721 !important;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .dashboard-card * {
            color: #333333 !important;
        }

        .sentiment-positive {
            color: #10B981 !important;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            background-color: #D1FAE5;
        }
        .sentiment-negative {
            color: #EF4444 !important;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            background-color: #FEE2E2;
        }
        .sentiment-neutral {
            color: #6B7280 !important;
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            background-color: #F3F4F6;
        }

        /* Strong targeting of input and textarea inside Streamlit */
        input[type="text"],
        textarea,
        .stTextInput > div > input,
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #cccccc !important;
        }

        /* Placeholder fix */
        input[type="text"]::placeholder,
        textarea::placeholder,
        .stTextInput > div > input::placeholder,
        .stTextArea textarea::placeholder {
            color: #555 !important;
        }

        /* Button styling */
        button {
            background-color: #1E3A8A !important;
            color: #ffffff !important;
            font-weight: bold !important;
            width: 80% !important;
            border-radius: 7px !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] button {
            color: #FFFFFF !important;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background-color: #1E3A8A;
            color: white !important;
        }

        button p {
            color: #ffffff !important;
        }

        .dataframe {
            color: #333333 !important;
        }

        .stMarkdown, p, li {
            color: #FFFFFF !important;
        }

        .dashboard-card .stMarkdown,
        .dashboard-card p,
        .dashboard-card li,
        .dashboard-card span,
        .dashboard-card div {
            color: #333333 !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
