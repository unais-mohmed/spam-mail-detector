import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only runs once, then cached)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))

# --- Same preprocessing function used during training ---
def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# --- Load the trained model and vectorizer ---
@st.cache_resource
def load_model():
    with open('spam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# --- Page layout ---
st.set_page_config(page_title="Spam Mail Detector", page_icon="📧")
st.title("📧 Spam Mail Detector")
st.write("Type or paste a message below to check if it's spam or not.")

message = st.text_area("Your message:", height=150, placeholder="e.g. Congratulations! You won a free prize...")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        clean_msg = preprocess(message)
        vec = vectorizer.transform([clean_msg])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]

        if prediction == 1:
            st.error(f"🚨 This looks like SPAM (confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ This looks like a normal message (confidence: {probability[0]*100:.1f}%)")

st.divider()
st.caption("Built with Naive Bayes + TF-IDF · Internship Project")