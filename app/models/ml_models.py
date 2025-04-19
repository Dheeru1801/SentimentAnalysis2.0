"""
Machine learning models for the Sentiment Analysis Dashboard.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Change to absolute import
from config import settings

@st.cache_resource
def train_model(data):
    """
    Train sentiment analysis models on the provided data.
    
    Args:
        data (DataFrame): DataFrame containing text data and sentiment labels
        
    Returns:
        dict: Dictionary containing trained models and evaluation metrics
    """
    # Clean data: drop rows with NaN in Summary or Sentiment and convert to string
    data = data.dropna(subset=['Summary', 'Sentiment'])
    
    # Ensure all text is string type
    data['Summary'] = data['Summary'].astype(str)
    
    # Extract features from the text data
    X = data['Summary'].values
    y = data['Sentiment'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize - make sure inputs are string type to avoid np.nan errors
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)
    
    # Train models
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_counts, y_train)
    
    rf_classifier = RandomForestClassifier(
        n_estimators=settings.RANDOM_FOREST_N_ESTIMATORS, 
        random_state=settings.RANDOM_FOREST_RANDOM_STATE
    )
    rf_classifier.fit(X_train_counts, y_train)
    
    # Evaluate
    y_pred_nb = nb_classifier.predict(X_test_counts)
    y_pred_rf = rf_classifier.predict(X_test_counts)
    
    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    
    nb_report = classification_report(y_test, y_pred_nb, output_dict=True)
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    
    return {
        'vectorizer': vectorizer,
        'nb_classifier': nb_classifier,
        'rf_classifier': rf_classifier,
        'nb_accuracy': nb_accuracy,
        'rf_accuracy': rf_accuracy,
        'nb_report': nb_report,
        'rf_report': rf_report
    }

def predict_sentiment(text, model_name, models):
    """
    Predict sentiment of text using the specified model.
    
    Args:
        text (str): Text to analyze
        model_name (str): Name of the model to use ('Naive Bayes' or 'Random Forest')
        models (dict): Dictionary containing trained models
        
    Returns:
        str: Predicted sentiment label
    """
    if not isinstance(text, str) or not text.strip():
        return "No text provided"
    
    text_counts = models['vectorizer'].transform([text])
    
    if model_name == "Naive Bayes":
        prediction = models['nb_classifier'].predict(text_counts)[0]
    else:  # Random Forest
        prediction = models['rf_classifier'].predict(text_counts)[0]
    
    return prediction