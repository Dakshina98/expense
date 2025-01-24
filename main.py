import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import os

# Load pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define categories and keywords
categories = {
    "Food": ["restaurant", "dining", "groceries", "coffee"],
    "Transport": ["taxi", "uber", "bus", "train", "flight"],
    "Utilities": ["electricity", "water", "internet", "gas"],
    "Shopping": ["clothes", "electronics", "furniture", "amazon"],
    "Entertainment": ["movies", "concert", "games", "subscription"],
}

# Function to categorize expenses
def categorize_expense(text):
    category_scores = {}
    for category, keywords in categories.items():
        embeddings = model.encode(keywords)
        text_embedding = model.encode(text)
        scores = [model.similarity(text_embedding, keyword_embedding) for keyword_embedding in embeddings]
        category_scores[category] = max(scores)

    # Return the category with the highest score
    return max(category_scores, key=category_scores.get)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Streamlit app
st.title("AI Expense Tracker")

# Upload receipt
uploaded_file = st.file_uploader("Upload a receipt (PDF or text)", type=["pdf", "txt"])

if uploaded_file:
    # Extract text from uploaded file
    if uploaded_file.type == "application/pdf":
        receipt_text = extract_text_from_pdf(uploaded_file)
    else:
        receipt_text = uploaded_file.read().decode("utf-8")
    
    st.subheader("Receipt Text:")
    st.write(receipt_text)

    # Categorize expenses
    category = categorize_expense(receipt_text)
    st.subheader("Predicted Category:")
    st.write(category)

    # Visualize spending
    st.subheader("Expense Summary:")
    fig, ax = plt.subplots()
    ax.bar(categories.keys(), [1 if cat == category else 0 for cat in categories.keys()])
    ax.set_ylabel("Count")
    ax.set_xlabel("Categories")
    st.pyplot(fig)
