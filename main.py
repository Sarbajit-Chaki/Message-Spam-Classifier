import joblib
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

text = input("Enter text: ")

# 1. Preprocessing
transformed_text = transform_text(text)

# 2. Vectorization
vectorized_text = vectorizer.transform([transformed_text])

# 3. Prediction
result = model.predict(vectorized_text)[0]


if result == 1:
    print("Spam")
else:
    print("Not Spam")