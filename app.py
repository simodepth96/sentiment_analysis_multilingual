import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
import io

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

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
st.title("Review Rating Distribution")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel file with a 'review' column", type=["xlsx"])

if uploaded_file:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)

    # Check if the 'review' column exists
    if 'review' not in df.columns:
        st.error("The uploaded file does not contain a 'review' column.")
    else:
        # Perform sentiment analysis
        st.info("Analyzing sentiment...")
        df['rating'] = df['review'].apply(analyze_sentiment)

        # Generate a pie chart
        rating_counts = df['rating'].value_counts().reset_index()
        rating_counts.columns = ['Rating', 'Count']
        fig = px.pie(
            rating_counts,
            values='Count',
            names='Rating',
            title="Review Rating Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig)

        # Save results to an Excel file
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        # Provide download button
        st.download_button(
            label="Download Rating Results",
            data=output,
            file_name="review_rating_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
