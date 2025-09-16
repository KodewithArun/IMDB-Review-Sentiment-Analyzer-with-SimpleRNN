import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Movie Review Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Clean CSS styling
st.markdown("""
<style>
    .main {
        background-color: #ffffff;
        padding: 2rem;
    }
    
    .main-title {
        color: #1f2937;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    .result-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    
    .positive-result {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #166534;
    }
    
    .negative-result {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #991b1b;
    }
    
    .sentiment-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .confidence-text {
        font-size: 1rem;
        color: #6b7280;
    }
    
    .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        margin-top: 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }
    
    .stTextArea > div > div > textarea {
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    .stSelectbox > div > div > select {
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    .progress-container {
        margin: 1.5rem 0;
    }
    
    .progress-bar {
        height: 8px;
        background-color: #e5e7eb;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        transition: width 0.3s ease;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .status-success {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MAX_FEATURES = 10000
MAX_LENGTH = 500
MODEL_PATH = "model/simplernn_model.h5"

@st.cache_resource
def load_model_and_data():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        word_index = imdb.get_word_index()
        return model, word_index
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_text(text, word_index):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=MAX_LENGTH)
    return padded

def predict_sentiment(text, model, word_index):
    processed_text = preprocess_text(text, word_index)
    prediction = model.predict(processed_text, verbose=0)[0][0]
    return float(prediction)

# Main App
st.markdown('<h1 class="main-title">🎬 Movie Review Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze the sentiment of movie reviews using AI</p>', unsafe_allow_html=True)

# Load model
model, word_index = load_model_and_data()

if model is None or word_index is None:
    st.error("❌ Failed to load model. Please check the model path.")
    st.stop()

# Success status
st.markdown('''
<div class="status-badge status-success">
    ✅ Model loaded successfully
</div>
''', unsafe_allow_html=True)

# Sample reviews
sample_reviews = [
    "Write your own review...",
    "This movie was absolutely fantastic! Great acting and amazing storyline.",
    "Terrible movie. Boring plot and poor acting. Complete waste of time.",
    "Average film with some good moments but nothing special overall.",
    "Masterpiece! One of the best movies I've ever seen. Highly recommend!"
]

# Input section
st.markdown("### Enter Movie Review")

selected_review = st.selectbox(
    "Choose a sample or write your own:",
    sample_reviews
)

if selected_review == "Write your own review...":
    review_text = st.text_area(
        "Your review:",
        height=120,
        placeholder="Enter your movie review here..."
    )
else:
    review_text = selected_review

# Analyze button
if st.button("Analyze Sentiment"):
    if review_text.strip():
        with st.spinner("Analyzing..."):
            score = predict_sentiment(review_text, model, word_index)
            
            # Determine sentiment
            is_positive = score > 0.5
            confidence = abs(score - 0.5) * 2
            
            # Results
            if is_positive:
                result_class = "result-card positive-result"
                sentiment_emoji = "😊"
                sentiment_label = "Positive"
                score_percent = f"{score:.1%}"
            else:
                result_class = "result-card negative-result"
                sentiment_emoji = "😞"
                sentiment_label = "Negative"
                score_percent = f"{(1-score):.1%}"
            
            # Display results
            st.markdown(f'''
            <div class="{result_class}">
                <div class="sentiment-text">
                    {sentiment_emoji} {sentiment_label} Review
                </div>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{score_percent}</div>
                        <div class="stat-label">Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{confidence:.0%}</div>
                        <div class="stat-label">Confidence</div>
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Progress bar
            st.markdown('<div class="progress-container">', unsafe_allow_html=True)
            st.progress(score, text=f"Sentiment Score: {score:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Celebration for very positive reviews
            if score > 0.8:
                st.balloons()
                st.success("🎉 Highly positive review detected!")
    else:
        st.warning("⚠️ Please enter a movie review to analyze.")

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #6b7280; font-size: 0.875rem;">'
    'Powered by TensorFlow and Streamlit</p>', 
    unsafe_allow_html=True
)