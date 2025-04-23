"""
Natural Language Processing utilities for the Sentiment Analysis Dashboard.
"""
import os
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk import pos_tag
nltk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "nltk_data"))
nltk.data.path.append(nltk_path)
# Change from relative to absolute import
from config import settings

nltk.data.path.append("app/utils/nltk_data")


# Download necessary NLTK data
# @st.cache_data
def download_nltk_data():
    """
    Download the necessary NLTK data resources.
    """
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    nltk.download('averaged_perceptron_tagger')
    
    # Fix for the specific error with averaged_perceptron_tagger_eng
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        # If the specific tagger isn't found, download it
        nltk.download('averaged_perceptron_tagger')
        
        # If still having issues, we'll use the default tagger
        global pos_tag
        pos_tag = nltk.tag.pos_tag  # Use default English tagger

# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def preprocess_text(text):
    """
    Clean and preprocess text for analysis.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Cleaned and preprocessed text
    """
    if isinstance(text, str):
        # Remove non-alphabetical characters and convert to lowercase
        clean_text = re.sub("[^a-zA-Z]", " ", text).lower().strip()
        return clean_text
    return ""

def extract_keywords(sentence):
    """
    Extract keywords (nouns and adjectives) from a sentence.
    
    Args:
        sentence (str): Input sentence
        
    Returns:
        str: Space-separated list of keywords
    """
    try:
        words = word_tokenize(sentence)
        # Try using the standard pos_tag function first
        try:
            pos_tags = pos_tag(words)
            keywords = [word for word, tag in pos_tags if tag in ('JJ', 'NN')]  # Adjectives and Nouns
        except LookupError:
            # If the specific tagger fails, use a simple keyword extraction approach
            # Just filter out common stopwords and return words
            stop_words = set(stopwords.words('english'))
            keywords = [word for word in words if word.lower() not in stop_words and len(word) > 3]
        
        return ' '.join(keywords)
    except Exception as e:
        st.warning(f"Issue with keyword extraction: {str(e)}")
        # Return the original words if there's any error
        return ' '.join([w for w in sentence.split() if len(w) > 3])

def analyze_sentiment_per_sentence(text):
    """
    Analyze sentiment of text by breaking it down into sentences.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        dict: Dictionary containing sentiment analysis results
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'sentence_sentiments': [],
            'positive_sentences': [],
            'negative_sentences': [],
            'contributing_keywords': {'positive': [], 'negative': []}
        }
        
    # Split the text into individual sentences
    sentences = nltk.sent_tokenize(text)
    
    # Dictionary to store sentiment results for each sentence
    sentence_sentiments = []
    
    # To track contributing sentences and keywords
    positive_sentences = []
    negative_sentences = []
    neutral_sentences = []
    contributing_keywords = {'positive': [], 'negative': [], 'neutral': []}
    
    for sentence in sentences:
        # Preprocess sentence
        clean_sentence = preprocess_text(sentence)
        
        # Get sentiment scores
        score = analyzer.polarity_scores(clean_sentence)
        
        # Classify sentiment based on score
        sentiment = "Neutral"
        if score['neg'] > settings.NEGATIVE_THRESHOLD:
            sentiment = "Negative"
            negative_sentences.append(sentence)
            contributing_keywords['negative'].extend(extract_keywords(sentence).split())
        elif score['pos'] > settings.POSITIVE_THRESHOLD:
            sentiment = "Positive"
            positive_sentences.append(sentence)
            contributing_keywords['positive'].extend(extract_keywords(sentence).split())
        else:
            neutral_sentences.append(sentence)
            contributing_keywords['neutral'].extend(extract_keywords(sentence).split())
        
        # Store sentence and its sentiment
        sentence_sentiments.append((sentence, sentiment, score))
    
    return {
        'sentence_sentiments': sentence_sentiments,
        'positive_sentences': positive_sentences,
        'negative_sentences': negative_sentences,
        'neutral_sentences': neutral_sentences,
        'contributing_keywords': contributing_keywords,
        'overall_sentiment': get_overall_sentiment([s[1] for s in sentence_sentiments])
    }

def get_overall_sentiment(sentiment_list):
    """
    Calculate the overall sentiment based on a list of sentence sentiments.
    
    Args:
        sentiment_list (list): List of sentiment labels
        
    Returns:
        str: Overall sentiment label
    """
    if not sentiment_list:
        return "Neutral"
        
    pos_count = sentiment_list.count("Positive")
    neg_count = sentiment_list.count("Negative")
    
    total = len(sentiment_list)
    
    if pos_count > neg_count and pos_count > total * 0.5:
        return "Positive"
    elif neg_count > pos_count and neg_count > total * 0.5:
        return "Negative"
    else:
        return "Neutral"
