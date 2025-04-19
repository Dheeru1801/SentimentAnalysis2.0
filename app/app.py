"""
Streamlit Sentiment Analysis Dashboard - Main Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import sys
from pathlib import Path

# Add the parent directory to the Python path to enable absolute imports
app_path = Path(__file__).parent
sys.path.insert(0, str(app_path))

# Import modules from refactored project structure
from config import settings
from utils.nlp_utils import download_nltk_data, analyze_sentiment_per_sentence
from utils.data_utils import load_data
from models.ml_models import train_model, predict_sentiment
from visualization.viz_utils import generate_wordcloud, load_css

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT,
    initial_sidebar_state=settings.INITIAL_SIDEBAR_STATE,
)

# Download NLTK data and load CSS
download_nltk_data()
load_css()

# App Title and Description
st.title("📊 Sentiment Analysis Dashboard")
st.markdown("""
<div class="dashboard-card">
This dashboard analyzes the sentiment of text reviews and comments. It uses both machine learning models 
(Naive Bayes and Random Forest) and NLTK's rule-based sentiment analyzer to determine if text expresses 
positive, negative, or neutral sentiment.
</div>
""", unsafe_allow_html=True)

# Load and prepare data
data = load_data()
models = train_model(data)

# Tabs for different sections
tabs = st.tabs(["✨ Analyze Text", "📈 Dataset Analysis", "📘 Model Performance", "📝 Batch Analysis"])

# Tab 1: Text Analysis
with tabs[0]:
    st.markdown("<h2>Sentiment Analysis Tool</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        user_input = st.text_area("Enter text to analyze:", 
                                  settings.SAMPLE_TEXT, 
                                  height=150)
        analyze_button = st.button("Analyze Sentiment", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        model_option = st.selectbox(
            "Choose analysis method:",
            ["NLTK Analyzer", "Naive Bayes", "Random Forest"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    if analyze_button and user_input:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        
        if model_option == "NLTK Analyzer":
            # NLTK Analysis
            analysis_result = analyze_sentiment_per_sentence(user_input)
            
            # Display overall sentiment
            overall_sentiment = analysis_result["overall_sentiment"]
            sentiment_class = f"sentiment-{overall_sentiment.lower()}"
            
            st.markdown(f"<h3>Overall Sentiment: <span class='{sentiment_class}'>{overall_sentiment}</span></h3>", 
                        unsafe_allow_html=True)
            
            # Display detailed analysis
            st.markdown("### Sentence-by-Sentence Analysis")
            
            for sentence, sentiment, score in analysis_result['sentence_sentiments']:
                sentiment_class = f"sentiment-{sentiment.lower()}"
                st.markdown(f"<p><b>Sentence:</b> {sentence}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Sentiment:</b> <span class='{sentiment_class}'>{sentiment}</span></p>", 
                            unsafe_allow_html=True)
                
                # Create a horizontal bar chart for sentiment scores
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=['Positive', 'Negative', 'Neutral'],
                    x=[score['pos'], score['neg'], score['neu']],
                    orientation='h',
                    marker=dict(
                        color=[settings.COLORS['Positive'], settings.COLORS['Negative'], settings.COLORS['Neutral']],
                        line=dict(color='rgba(0, 0, 0, 0.2)', width=1)
                    )
                ))
                fig.update_layout(
                    title=f"Sentiment Scores",
                    xaxis_title="Score",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("<hr>", unsafe_allow_html=True)
            
            # Display contributing keywords
            col1, col2 = st.columns(2)
            
            with col1:
                if analysis_result['contributing_keywords']['positive']:
                    st.markdown("### Positive Keywords")
                    pos_keywords = " ".join(analysis_result['contributing_keywords']['positive'])
                    if pos_keywords.strip():
                        fig = generate_wordcloud(pos_keywords, "Positive Keywords")
                        st.pyplot(fig)
            
            with col2:
                if analysis_result['contributing_keywords']['negative']:
                    st.markdown("### Negative Keywords")
                    neg_keywords = " ".join(analysis_result['contributing_keywords']['negative'])
                    if neg_keywords.strip():
                        fig = generate_wordcloud(neg_keywords, "Negative Keywords")
                        st.pyplot(fig)
        else:
            # ML model prediction
            prediction = predict_sentiment(user_input, model_option, models)
            sentiment_class = f"sentiment-{prediction.lower()}"
            
            st.markdown(f"<h3>Predicted Sentiment: <span class='{sentiment_class}'>{prediction}</span></h3>", 
                        unsafe_allow_html=True)
            
            # Also show NLTK analysis for comparison
            st.markdown("### NLTK Analysis (For Comparison)")
            analysis_result = analyze_sentiment_per_sentence(user_input)
            
            for sentence, sentiment, score in analysis_result['sentence_sentiments']:
                sentiment_class = f"sentiment-{sentiment.lower()}"
                st.markdown(f"<p><b>Sentence:</b> {sentence}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Sentiment:</b> <span class='{sentiment_class}'>{sentiment}</span></p>", 
                            unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Dataset Analysis
with tabs[1]:
    st.markdown("<h2>Dataset Overview</h2>", unsafe_allow_html=True)
    
    # Display sample of the dataset
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.write("### Sample Data")
    st.dataframe(data.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Dataset statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.write("### Dataset Statistics")
        st.write(f"Total reviews: {len(data):,}")
        
        # Count sentiments
        sentiment_counts = data['Sentiment'].value_counts()
        
        # Create a pie chart for sentiment distribution
        fig = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map=settings.COLORS
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.write("### Review Text Word Cloud")
        
        # Generate word cloud for all reviews
        all_text = " ".join(data['Summary'].dropna())
        fig = generate_wordcloud(all_text)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sentiment by category (if available)
    if 'categories' in data.columns:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment by Category")
        
        # Get the top categories
        categories = []
        for cat_list in data['categories'].dropna().str.split(','):
            if isinstance(cat_list, list):
                categories.extend(cat_list)
        
        top_categories = pd.Series(categories).value_counts().head(10).index.tolist()
        
        # Filter data for top categories
        category_data = []
        for category in top_categories:
            cat_positive = len(data[(data['categories'].str.contains(category, na=False)) & 
                                   (data['Sentiment'] == 'Positive')])
            cat_negative = len(data[(data['categories'].str.contains(category, na=False)) & 
                                   (data['Sentiment'] == 'Negative')])
            cat_neutral = len(data[(data['categories'].str.contains(category, na=False)) & 
                                  (data['Sentiment'] == 'Neutral')])
            
            category_data.append({
                'Category': category,
                'Positive': cat_positive,
                'Negative': cat_negative,
                'Neutral': cat_neutral
            })
        
        category_df = pd.DataFrame(category_data)
        
        # Create grouped bar chart
        fig = px.bar(
            category_df.melt(id_vars=['Category'], var_name='Sentiment', value_name='Count'),
            x='Category',
            y='Count',
            color='Sentiment',
            title='Sentiment Distribution by Top Categories',
            color_discrete_map=settings.COLORS
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Model Performance
with tabs[2]:
    st.markdown("<h2>Model Performance</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.write("### Naive Bayes Classifier")
        st.metric("Accuracy", f"{models['nb_accuracy']:.2%}")
        
        # Create classification report table
        nb_report = models['nb_report']
        nb_df = pd.DataFrame({
            'Precision': [nb_report[c]['precision'] for c in ['Positive', 'Negative', 'Neutral'] if c in nb_report],
            'Recall': [nb_report[c]['recall'] for c in ['Positive', 'Negative', 'Neutral'] if c in nb_report],
            'F1-Score': [nb_report[c]['f1-score'] for c in ['Positive', 'Negative', 'Neutral'] if c in nb_report]
        }, index=[c for c in ['Positive', 'Negative', 'Neutral'] if c in nb_report])
        
        st.table(nb_df.style.format("{:.2%}"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.write("### Random Forest Classifier")
        st.metric("Accuracy", f"{models['rf_accuracy']:.2%}")
        
        # Create classification report table
        rf_report = models['rf_report']
        rf_df = pd.DataFrame({
            'Precision': [rf_report[c]['precision'] for c in ['Positive', 'Negative', 'Neutral'] if c in rf_report],
            'Recall': [rf_report[c]['recall'] for c in ['Positive', 'Negative', 'Neutral'] if c in rf_report],
            'F1-Score': [rf_report[c]['f1-score'] for c in ['Positive', 'Negative', 'Neutral'] if c in rf_report]
        }, index=[c for c in ['Positive', 'Negative', 'Neutral'] if c in rf_report])
        
        st.table(rf_df.style.format("{:.2%}"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model comparison
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown("### Model Comparison")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    nb_metrics = [models['nb_accuracy']]
    rf_metrics = [models['rf_accuracy']]
    
    # Add weighted averages for precision, recall, f1
    for metric in ['precision', 'recall', 'f1-score']:
        nb_metrics.append(models['nb_report']['weighted avg'][metric])
        rf_metrics.append(models['rf_report']['weighted avg'][metric])
    
    comparison_df = pd.DataFrame({
        'Metric': metrics,
        'Naive Bayes': nb_metrics,
        'Random Forest': rf_metrics
    })
    
    fig = px.bar(
        comparison_df.melt(id_vars=['Metric'], var_name='Model', value_name='Score'),
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        title='Model Performance Comparison',
        text_auto='.2%'
    )
    fig.update_layout(yaxis_range=[0, 1])
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Batch Analysis
with tabs[3]:
    st.markdown("<h2>Batch Sentiment Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.write("Upload a CSV file with a column named 'text' or 'review' to analyze multiple texts at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find the text column
            text_col = None
            for col in ['text', 'review', 'comment', 'Review', 'Text', 'Comment', 'reviews_text', 'Summary']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                st.error("Could not find a column with text to analyze. Please ensure your CSV has a column named 'text', 'review', or 'comment'.")
            else:
                # Choose model for batch analysis
                model_option = st.selectbox(
                    "Choose analysis method for batch processing:",
                    ["NLTK Analyzer", "Naive Bayes", "Random Forest"]
                )
                
                if st.button("Analyze Batch", use_container_width=True):
                    # Process the data
                    with st.spinner('Analyzing texts...'):
                        if model_option == "NLTK Analyzer":
                            # Apply NLTK analysis
                            results = []
                            for text in df[text_col]:
                                if isinstance(text, str) and text.strip():
                                    analysis = analyze_sentiment_per_sentence(text)
                                    results.append(analysis["overall_sentiment"])
                                else:
                                    results.append("Neutral")
                            
                            df['Sentiment'] = results
                        else:
                            # Apply ML model
                            results = []
                            for text in df[text_col]:
                                if isinstance(text, str) and text.strip():
                                    prediction = predict_sentiment(text, model_option, models)
                                    results.append(prediction)
                                else:
                                    results.append("Neutral")
                            
                            df['Sentiment'] = results
                        
                        # Display results
                        sentiment_counts = df['Sentiment'].value_counts()
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write("### Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Create download link
                            csv = df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_analysis_results.csv">Download Results as CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        
                        with col2:
                            st.write("### Sentiment Distribution")
                            
                            # Create pie chart for sentiment distribution
                            fig = px.pie(
                                values=sentiment_counts.values, 
                                names=sentiment_counts.index,
                                title="Sentiment Distribution",
                                color=sentiment_counts.index,
                                color_discrete_map=settings.COLORS
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 30px'>
    <p style='color: #666; font-size: 14px;'>Sentiment Analysis Dashboard • Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)