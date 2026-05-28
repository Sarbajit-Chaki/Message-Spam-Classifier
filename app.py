import streamlit as st
import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y.copy() 
    y.clear()
    stop_words = set(stopwords.words('english'))
    for word in text:
        if word not in stop_words and word not in string.punctuation:
            y.append(word)

    text = y.copy()
    y.clear()
    for word in text:
        y.append(ps.stem(word))

    text = y.copy()
    y.clear()
    
    return " ".join(text)

# Load trained model
model = joblib.load('model.pkl')

# Load vectorizer
vectorizer = joblib.load('vectorizer.pkl')


# Page setup
st.set_page_config(page_title="Spam Detection System", page_icon="🚫", layout="centered")

# Main container
st.markdown('<div style="font-size: 2.5rem; margin-left:-7px; margin-bottom:20px; font-weight: bold;">📩 Spam Detection Classifier</div>', unsafe_allow_html=True)

# Input
text = st.text_area('message', placeholder='Enter your message...', label_visibility='collapsed', height=120)

# Button
if st.button("Predict"):
    if not text.strip():
        st.warning("⚠️ Please enter a message before detection.")
    elif len(text) < 10:
        st.warning("⚠️ Message is too short for detection.")
    else:
        # 1. Preprocessing
        transformed_text = transform_text(text)

        # 2. Vectorization
        vectorized_text = vectorizer.transform([transformed_text])

        # 3. Prediction
        result = model.predict(vectorized_text)[0]

        result = "spam" if result == 1 else "not spam"
        
        color = "#ef4444" if result == "spam" else "#22c55e"
        st.markdown(
            f'<div style="text-align: center; font-weight: bold; font-size: 20px; border-radius: 5px; padding: 4px 0px; background-color:{color}; color:white;">{result}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)