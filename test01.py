
# -*- coding: utf-8 -*-

import pandas as pd
from transformers import pipeline
import streamlit as st

# Load the dataset
dataset_path = "car-reviews.csv"  # replace with the actual path
df = pd.read_csv(dataset_path)

# Display the dataset
st.dataframe(df)

# Preprocess the dataset (assuming the dataset columns are named as mentioned)
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime format
df.set_index('Date', inplace=True)  # Set 'Date' as the index for time-based queries

# Load RAG model
rag_model = pipeline('text2text-generation', model='facebook/rag-token-base', device=0)  # You can replace with other models

# Define a function to search for reviews based on user queries
def search_reviews(model, model_name, year):
    query = f"What are the reviews regarding {model_name} in year {year}?"
    response = model(query, max_length=200, num_return_sequences=1)
    return response[0]['generated']['text']

# Streamlit interface
st.title("Auto Review Search")

# User input for model and year
model_name = st.text_input("Enter Suzuki Model:")
year = st.number_input("Enter Year:", min_value=int(df.index.year.min()), max_value=int(df.index.year.max()))

# Button to trigger the search
if st.button("Search"):
    # Search for reviews
    result = search_reviews(rag_model, model_name, year)
    
    # Display the result
    st.subheader("Review:")
    st.text(result)
