import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import io
import plotly.express as px

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Perform sentiment analysis
def analyze_sentiment(review):
    inputs = tokenizer.encode_plus(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted_class = torch.argmax(scores).item()
    return predicted_class + 1  # Ratings are 1-indexed

# Streamlit app
st.title("Sentiment Analysis on Reviews")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read the uploaded file into a dataframe
    df_2 = pd.read_excel(uploaded_file)

    # Check if the 'review' column exists
    if 'review' not in df_2.columns:
        st.error("The uploaded file does not contain a 'Review' column.")
    else:
        # Perform sentiment analysis on the reviews
        df_2['sentiment'] = df_2['review'].apply(analyze_sentiment)

        #prep a pie chart
        fig = px.pie(df_2, names='sentiment', title='Sentiment Analysis Results')

        # Show the plot in Streamlit
        st.pyplot(fig)

        # Save the results to an Excel buffer for download
        excel_buffer = io.BytesIO()
        df_2.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)  # Move to the start of the buffer

        # Then in your Streamlit download button
        st.download_button(
            label="Download Excel file",
            data=excel_buffer.getvalue(),
            file_name="sentiment_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
