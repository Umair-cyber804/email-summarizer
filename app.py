# ==========================================================
# Smart Email Summarizer
# AI Powered Email Summarization using DistilBART
# ==========================================================

# ==========================================================
# Import Libraries with Fallbacks
# ==========================================================

import streamlit as st
import subprocess
import sys
import os

# Auto-install missing packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try importing torch, install if missing
try:
    import torch
except ImportError:
    st.warning("Installing PyTorch... This may take a moment.")
    install_package("torch")
    import torch

try:
    import transformers
except ImportError:
    st.warning("Installing Transformers...")
    install_package("transformers")
    import transformers

try:
    import PyPDF2
except ImportError:
    install_package("PyPDF2")
    import PyPDF2

try:
    import spacy
except ImportError:
    install_package("spacy")
    import spacy

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Smart Email Summarizer",
    page_icon="📧",
    layout="wide"
)

# ==========================================================
# Download spaCy Model
# ==========================================================

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.info("Downloading spaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ==========================================================
# Load DistilBART Model
# ==========================================================

@st.cache_resource
def load_model():
    # Use Hugging Face model directly
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(
        model_name,
        low_cpu_mem_usage=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==========================================================
# Load Sentiment Analysis Model
# ==========================================================

@st.cache_resource
def load_sentiment_model():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    return sentiment

sentiment_model = load_sentiment_model()

# ==========================================================
# Email Summarization Function
# ==========================================================

def summarize_email(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=30,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )
    return summary

# ==========================================================
# Keyword Extraction Function
# ==========================================================

def extract_keywords(text):
    doc = nlp(text)
    keywords = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            word = token.text.lower()
            if len(word) > 2:
                if word not in keywords:
                    keywords.append(word)

    return keywords[:8]

# ==========================================================
# Sentiment Analysis Function
# ==========================================================

def analyze_sentiment(text):
    result = sentiment_model(text[:512])
    return result[0]

# ==========================================================
# Named Entity Recognition (NER)
# ==========================================================

def extract_entities(text):
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        entities.append((ent.text, ent.label_))

    return entities

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:
    st.title("📧 Smart Email Summarizer")
    st.markdown("---")
    st.subheader("About Project")
    st.write("""
This AI application summarizes long emails using
the DistilBART Transformer model.

### Features

• AI Email Summarization
• TXT/PDF Upload
• Keyword Extraction
• Download Summary
• Email Statistics
• Built using Streamlit
""")
    st.markdown("---")
    st.subheader("Technology Stack")
    st.write("🐍 Python")
    st.write("🤖 Hugging Face")
    st.write("📄 DistilBART")
    st.write("🧠 spaCy")
    st.write("🎨 Streamlit")

# ==========================================================
# Main Page
# ==========================================================

st.title("📧 Smart Email Summarizer")
st.write("""
Generate concise summaries of long emails using the
**DistilBART Transformer Model**.
""")
st.markdown("---")

# ==========================================================
# Upload Email File
# ==========================================================

uploaded_file = st.file_uploader(
    "📂 Upload Email (.txt or .pdf)",
    type=["txt", "pdf"]
)

email = ""

# ==========================================================
# Read Uploaded File
# ==========================================================

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        email = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                email += text

# ==========================================================
# Email Text Area
# ==========================================================

email = st.text_area(
    "📨 Or Paste Your Email Here",
    value=email,
    height=250,
    placeholder="Paste your email here..."
)

# ==========================================================
# Email Statistics
# ==========================================================

if email.strip():
    words = len(email.split())
    characters = len(email)
    reading_time = round(words / 200, 2)

    st.subheader("📊 Email Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📝 Words", words)
    with col2:
        st.metric("🔠 Characters", characters)
    with col3:
        st.metric("⏱ Reading Time", f"{reading_time} min")

st.markdown("---")

# ==========================================================
# Generate Summary Button
# ==========================================================

if st.button("🚀 Generate Summary"):
    if email.strip() == "":
        st.warning("⚠️ Please upload or paste an email.")
    else:
        with st.spinner("Generating AI Summary..."):
            summary = summarize_email(email)

        st.success("✅ Summary Generated Successfully!")
        st.subheader("📄 AI Generated Summary")
        st.info(summary)

        # ==========================================================
        # Keyword Extraction
        # ==========================================================
        st.markdown("---")
        st.subheader("🏷️ Extracted Keywords")
        keywords = extract_keywords(email)

        if len(keywords) > 0:
            cols = st.columns(4)
            for i, keyword in enumerate(keywords):
                cols[i % 4].success(keyword)
        else:
            st.info("No keywords found.")

        # ==========================================================
        # Sentiment Analysis
        # ==========================================================
        st.markdown("---")
        st.subheader("😊 Email Sentiment")
        sentiment = analyze_sentiment(email)
        label = sentiment["label"]
        score = sentiment["score"]

        if label == "POSITIVE":
            st.success(f"😊 Positive ({score:.2%} confidence)")
        else:
            st.error(f"😞 Negative ({score:.2%} confidence)")

        # ==========================================================
        # Named Entity Recognition
        # ==========================================================
        st.markdown("---")
        st.subheader("👤 Named Entities")
        entities = extract_entities(email)

        if entities:
            for entity, label in entities:
                st.write(f"**{entity}** → {label}")
        else:
            st.info("No named entities found.")

        # ==========================================================
        # Summary Analytics
        # ==========================================================
        st.markdown("---")
        st.subheader("📊 Summary Analytics")

        original_words = len(email.split())
        summary_words = len(summary.split())
        compression = round((1 - (summary_words / original_words)) * 100, 2)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Original Words", original_words)
        with col2:
            st.metric("Summary Words", summary_words)
        with col3:
            st.metric("Compression", f"{compression}%")

        # ==========================================================
        # Download Summary
        # ==========================================================
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name="summary.txt"
        )