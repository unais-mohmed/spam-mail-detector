# 📩 Spam Mail Detector

A machine learning web app that classifies text messages as **Spam** or **Ham** (not spam), built with Python, scikit-learn, and Streamlit.

## Overview

This project was built as part of my AI/ML internship (July–August 2026). It uses Natural Language Processing (NLP) techniques to analyze SMS text messages and predict whether they are spam.

## Features

- Text preprocessing (lowercasing, tokenization, stopword removal)
- TF-IDF vectorization for converting text into numerical features
- Naive Bayes classification model
- Interactive Streamlit web interface with confidence scores
- Achieves ~96% accuracy on test data

## Tech Stack

- Python
- scikit-learn
- NLTK
- Streamlit
- Pandas

## Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data) — 5,574 labeled SMS messages (ham/spam).

## How It Works

1. Load and clean the raw text data
2. Preprocess text (lowercase, tokenize, remove stopwords)
3. Convert text to numeric features using TF-IDF
4. Train a Multinomial Naive Bayes classifier
5. Evaluate using accuracy, precision, and F1 score
6. Serve predictions through a Streamlit web app

## Results

- **Accuracy:** ~96%
- **Precision:** 1.00
- The model prioritizes precision over recall — it rarely misclassifies real messages as spam, at the small cost of missing a few borderline spam messages.

## How to Run Locally

\`\`\`bash
git clone https://github.com/unais-mohemd/spam-mail-detector.git
cd spam-mail-detector
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Author

@unais-mohmed
