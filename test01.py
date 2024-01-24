# -*- coding: utf-8 -*-

import pandas as pd
from transformers import pipeline
import streamlit as st

dataset_path = "car-reviews.csv"
df = pd.read_csv(dataset_path)

st.dataframe(df)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

rag_model = pipeline('text2text-generation', model='facebook/rag-token-base', device=0)

def search_reviews(model, model_name, year):
    query = f"What are the reviews regarding {model_name} in year {year}?"
    response = model(query, max_length=200, num_return_sequences=1)
    return response[0]['generated']['text']

st.title("Auto Review Search")

model_name = st.text_input("Enter Suzuki Model:")
year = st.number_input("Enter Year:", min_value=int(df.index.year.min()), max_value=int(df.index.year.max()))

if st.button("Search"):
    result = search_reviews(rag_model, model_name, year)
    
    st.subheader("Review:")
    st.text(result)
