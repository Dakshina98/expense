import streamlit as st
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np

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

# Streamlit app
st.title("AI Expense Tracker")

# Upload receipt (TXT file)
uploaded_file = st.file_uploader("Upload a receipt (TXT format)", type=["txt"])

if uploaded_file:
    # Read the receipt text line by line
    receipt_text = uploaded_file.read().decode("utf-8").strip()
    expense_lines = receipt_text.split("\n")  # Split into lines

    # Display receipt text
    st.subheader("Receipt Details:")
    st.write(receipt_text)

    # Categorize each expense and count categories
    category_counts = {category: 0 for category in categories.keys()}
    for line in expense_lines:
        category = categorize_expense(line)
        category_counts[category] += 1

    # Display category counts
    st.subheader("Categorized Expenses:")
    for category, count in category_counts.items():
        st.write(f"{category}: {count}")

    # Visualize spending as a bar chart
    st.subheader("Expense Summary (Categories vs Count):")
    fig, ax = plt.subplots()
    ax.bar(category_counts.keys(), category_counts.values(), color='skyblue')
    ax.set_ylabel("Count")
    ax.set_xlabel("Categories")
    ax.set_title("Expense Summary")
    plt.xticks(rotation=45)
    st.pyplot(fig)
