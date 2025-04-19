# Sentiment Analysis Dashboard

A comprehensive web-based dashboard for analyzing sentiment in text reviews and comments using both machine learning models and rule-based approaches.

![Sentiment Analysis Dashboard](https://img.shields.io/badge/Streamlit-Sentiment%20Analysis-ff4b4b)

## Overview

This Sentiment Analysis Dashboard is a full-featured web application that analyzes text reviews and comments to determine whether they express positive, negative, or neutral sentiment. It provides detailed analysis including sentence-by-sentence breakdowns, contributing keywords, and various visualizations to help understand sentiment patterns.

## Features

- **Multiple Analysis Methods**: Choose between NLTK's VADER analyzer, Naive Bayes classifier, and Random Forest classifier
- **Detailed Text Analysis**: Sentence-by-sentence sentiment breakdown with scores
- **Keyword Extraction**: Identification of words that contribute to positive or negative sentiment
- **Data Visualization**: Word clouds, bar charts, and pie charts for sentiment distribution
- **Model Performance Comparison**: Detailed metrics to compare different ML models
- **Batch Processing**: Analyze multiple texts at once via CSV upload
- **Dark Mode UI**: Modern interface with proper text contrast
- **Dataset Exploration**: Built-in visualization of the dataset statistics

## Project Structure

```
ReviewsSentimentAnalysis/
├── Dataset-SA.csv           # Main dataset for sentiment analysis
├── demo_dataset.csv         # Sample dataset for demonstrations
├── test_data.csv            # Test dataset
├── train_data.csv           # Training dataset
├── README.md                # Project documentation
└── app/                     # Main application directory
    ├── __init__.py          # Makes app a Python package
    ├── app.py               # Main Streamlit application
    ├── assets/              # Static files
    │   └── style.css        # Additional CSS styles (if needed)
    ├── config/              # Configuration settings
    │   ├── __init__.py
    │   └── settings.py      # Application settings and constants
    ├── models/              # Machine learning model components
    │   ├── __init__.py
    │   └── ml_models.py     # ML model training and prediction functions
    ├── utils/               # Utility functions
    │   ├── __init__.py
    │   ├── data_utils.py    # Data loading and processing
    │   └── nlp_utils.py     # NLP processing utilities
    └── visualization/       # Visualization components
        ├── __init__.py
        └── viz_utils.py     # Functions for generating charts and visualizations
```

## Component Details

### Main Application (`app/app.py`)

The entry point of the application. It:

- Sets up the Streamlit interface
- Orchestrates the workflow by importing and using components from other modules
- Creates the interactive UI with tabs for different functionalities
- Manages user inputs and displays results

The application has four main tabs:

1. **Analyze Text**: For analyzing individual text inputs
2. **Dataset Analysis**: For exploring the dataset statistics and visualizations
3. **Model Performance**: For comparing different ML models' performance
4. **Batch Analysis**: For analyzing multiple texts at once via CSV upload

### Configuration (`app/config/settings.py`)

Contains:

- Constants and settings used throughout the application
- Page configuration parameters
- Model settings like thresholds and parameters
- Visualization settings
- Color schemes and other UI constants
- Helper functions for file paths

### Models (`app/models/ml_models.py`)

Handles all machine learning functionality:

- Training models (Naive Bayes and Random Forest classifiers)
- Model evaluation and performance metrics
- Sentiment prediction functions
- Text vectorization using CountVectorizer

### NLP Utilities (`app/utils/nlp_utils.py`)

Contains natural language processing functions:

- NLTK data downloading and setup
- Text preprocessing functions
- Sentiment analysis using NLTK's VADER analyzer
- Keyword extraction from text
- Sentence-by-sentence sentiment analysis
- Overall sentiment calculation

### Data Utilities (`app/utils/data_utils.py`)

Manages data operations:

- Loading datasets from files
- Data preprocessing
- Handling missing values
- Providing sample data when needed

### Visualization (`app/visualization/viz_utils.py`)

Contains:

- WordCloud generation for visualizing keyword frequencies
- Fallback visualizations when WordCloud fails (bar charts)
- CSS styling for the Streamlit interface
- Chart formatting and color schemes

## Data Files

- `Dataset-SA.csv`: The main sentiment analysis dataset
- `demo_dataset.csv`: A smaller dataset for demonstrations
- `train_data.csv` and `test_data.csv`: Split datasets for training and testing models

## Requirements

The application requires the following Python packages:

- streamlit
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- plotly
- wordcloud

## Installation



1. Install the required packages:

```bash
pip install streamlit pandas numpy scikit-learn nltk matplotlib plotly wordcloud
```

2. Download NLTK data (the application will also do this automatically on first run):

```bash
python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

## Running the Application

To start the Sentiment Analysis Dashboard:

```bash
cd ReviewsSentimentAnalysis
streamlit run app/app.py
```

This will launch the application and open it in your default web browser. If it doesn't open automatically, you can access it at http://localhost:8501.

## Usage Guide

### 1. Text Analysis Tab

- Enter text in the text area
- Choose your preferred analysis method (NLTK, Naive Bayes, or Random Forest)
- Click "Analyze Sentiment" to see results
- View overall sentiment, sentence-by-sentence breakdown, and contributing keywords

### 2. Dataset Analysis Tab

- Browse dataset statistics and distribution
- View word clouds of common terms in the dataset
- Explore sentiment distribution across different categories (if available)

### 3. Model Performance Tab

- Compare accuracy, precision, recall, and F1-score across models
- View detailed classification reports for each model
- Analyze performance metrics through interactive charts

### 4. Batch Analysis Tab

- Upload a CSV file with a column containing text to analyze
- Choose your preferred analysis method
- Process multiple texts at once
- Download results as CSV
- View sentiment distribution of the batch

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NLTK and VADER for sentiment analysis capabilities
- Streamlit for the web application framework
- Scikit-learn for machine learning models
