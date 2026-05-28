# 📩 Message Spam Classifier

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam**. This project goes beyond a basic classifier — it includes thorough **Exploratory Data Analysis (EDA)**, a complete **NLP preprocessing pipeline**, a **multi-model comparison** across 7 algorithms, and a deployed **Streamlit web app**.

---

## 🧠 Model Overview

| Property        | Detail                              |
|-----------------|-------------------------------------|
| Algorithm       | Multinomial Naive Bayes (MultinomialNB) |
| Model Type      | Supervised Learning — Binary Classification |
| Dataset         | UCI SMS Spam Collection (`spam.csv`) |
| Dataset Size    | ~5,169 messages (after deduplication) |
| Labels          | `spam` (1) / `ham` → `not spam` (0) |
| Vectorization   | TF-IDF (`TfidfVectorizer`, max 3000 features) |
| Train/Test Split | 80% / 20% (random_state=42) |

---

## 🗂️ Project Structure

```
Message-Spam-Classifier/
│
├── sms-spam-detection.ipynb  # Full ML notebook (EDA → preprocessing → training → evaluation)
├── app.py                    # Streamlit web app
├── main.py                   # CLI inference script
├── model.pkl                 # Serialized MultinomialNB model
├── vectorizer.pkl            # Serialized TF-IDF vectorizer
├── spam.csv                  # UCI SMS Spam Collection dataset
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🔬 Notebook Walkthrough

### 1. 📥 Data Loading & Cleaning

- Loaded `spam.csv` (latin1 encoding) using `pandas`
- Dropped 3 unnamed/irrelevant columns (`Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`)
- Renamed columns: `v1` → `spam`, `v2` → `text`
- Encoded labels using `LabelEncoder`: `ham` → `0`, `spam` → `1`
- Checked for and removed **duplicate rows** (`keep='first'`)
- Verified no missing values

---

### 2. 📊 Exploratory Data Analysis (EDA)

**Class Distribution:**
- Dataset is **imbalanced** — significantly more ham than spam messages
- Visualized with a pie chart

**Feature Engineering — 3 new statistical columns added per message:**

| Feature          | Description                          |
|------------------|--------------------------------------|
| `num_characters` | Total character count                |
| `num_words`      | Word count (via NLTK tokenizer)      |
| `num_sentences`  | Sentence count (via NLTK tokenizer)  |

**Key Insights from EDA:**
- Spam messages tend to be **significantly longer** (more characters, words, sentences) than ham messages
- `num_characters` has the **highest correlation** with the target label
- High inter-correlation among the 3 feature columns was noted (expected, since longer texts have more words and sentences)
- Distribution plots (histplot with `stat='density'`) used to fairly compare imbalanced classes

**Word Cloud Analysis:**
- Generated word clouds separately for spam and ham messages after preprocessing to visually surface most frequent terms

**Top 30 Frequent Words:**
- Bar plots created for both spam and ham corpora to understand class-specific vocabulary patterns

---

### 3. ⚙️ NLP Preprocessing Pipeline

Each message is passed through the `transform_text()` function:

```
Raw Text
   ↓ Lowercase
   ↓ Tokenize (NLTK word_tokenize)
   ↓ Keep alphanumeric tokens only
   ↓ Remove English stopwords (NLTK)
   ↓ Remove punctuation
   ↓ Stem each word (Porter Stemmer)
   ↓ Join back to string
```

**Libraries used:** `nltk`, `string`, `PorterStemmer`, `word_tokenize`, `stopwords`

The cleaned text is stored in a new column `transformed_text` and used for all subsequent modeling steps.

---

### 4. 🔢 Vectorization

- Used **TF-IDF** (`TfidfVectorizer`) with `max_features=3000`
- TF-IDF was chosen over CountVectorizer for its ability to down-weight common terms and improve precision
- Output shape: `(5169, 3000)` — 5,169 messages × 3,000 unique features

---

### 5. 🤖 Model Building & Comparison

Three Naive Bayes variants were first evaluated:

| Model         | Notes                                     |
|---------------|-------------------------------------------|
| GaussianNB    | Baseline NB variant                       |
| MultinomialNB | Best suited for text/count data           |
| BernoulliNB   | Binary presence/absence of terms         |

**Key finding:** TF-IDF vectorization significantly improved **precision** for both MultinomialNB and BernoulliNB over CountVectorizer.

Then a **broader comparison of 7 classifiers** was run:

| Algorithm            |
|----------------------|
| Logistic Regression  |
| SVC                  |
| Multinomial NB       |
| K-Nearest Neighbors  |
| Decision Tree        |
| Random Forest        |
| Gradient Boosting    |

Results were compiled into a `performance_df` DataFrame and visualized using a grouped `catplot` (accuracy + precision per model), sorted by precision.

**MultinomialNB** was selected as the final model for its **best precision score** — minimizing false positives (legitimate messages incorrectly flagged as spam) is the priority metric for this use case.

---

### 6. 💾 Model Serialization

```python
joblib.dump(tfidf, 'vectorizer.pkl')
joblib.dump(mnb, 'model.pkl')
```

Both the trained vectorizer and model are saved for reuse in `app.py` and `main.py` without retraining.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sarbajit-Chaki/Message-Spam-Classifier.git
cd Message-Spam-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK data (first run only)

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### 4. Run the Streamlit web app

```bash
streamlit run app.py
```

### 5. Or use the CLI

```bash
python main.py
```

---

## 🛠️ Tech Stack

- **Language**: Python 🐍
- **Data**: `pandas`
- **NLP**: `nltk` (tokenization, stopwords, Porter Stemmer)
- **ML**: `scikit-learn` (TF-IDF, MultinomialNB, model comparison)
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
- **Serialization**: `joblib`
- **Web App**: `Streamlit`

---


## 📄 License

Feel free to contribute, modify, or use this project according to the terms of the license.
