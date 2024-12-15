import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import plotly.express as px
import streamlit as st
import io

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Define sentiment analysis function
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

# Define function to classify emotions
def calculate_emotion(smag):
    if smag > 0 and smag < 2:
        return "Poor"
    elif smag >= 5:
        return "Positive"
    elif smag > 3 and smag < 4:
        return "Neutral"
    elif smag >= 2 and smag <= 3:
        return "Poor"
    elif smag >= 4 and smag < 5:
        return "Neutral"
    else:
        return "Undefined"

# Streamlit App
st.title("Sentiment Analysis on Product Reviews")

# File upload widget
uploaded_file = st.file_uploader("Upload your Excel file with a 'review' column", type=["xlsx"])

if uploaded_file:
    # Read the uploaded Excel file
    df_2 = pd.read_excel(uploaded_file)

    # Check if the 'review' column exists
    if 'review' not in df_2.columns:
        st.error("The uploaded file does not contain a 'review' column.")
    else:
        # Perform sentiment analysis
        st.info("Analyzing sentiment...")
        df_2['sentiment'] = df_2['review'].apply(analyze_sentiment)
        df_2['emotion'] = df_2['sentiment'].apply(calculate_emotion)

        # Display results
        st.write("Sentiment Analysis Results", df_2.head())

        # Pie chart for emotion distribution
        emotion_counts = df_2['emotion'].value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        fig_pie = px.pie(
            emotion_counts,
            values='Count',
            names='Emotion',
            title="Emotion Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie)

        # Scatter plot for sentiment distribution
        fig_scatter = px.scatter(
            x=df_2['sentiment'],
            y=[0] * len(df_2),  # Zero y-axis for scatterplot
            color=df_2['emotion'],
            title="Sentiment Magnitude Distribution",
            labels={'x': "Sentiment Score", 'y': ""},
            color_discrete_map={"Poor": "red", "Neutral": "yellow", "Positive": "green", "Undefined": "gray"}
        )
        fig_scatter.update_layout(yaxis_visible=False)
        st.plotly_chart(fig_scatter)

        # Save results to an Excel file
        output = io.BytesIO()
        df_2.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        # Provide download button for the results
        st.download_button(
            label="Download Sentiment Analysis Results",
            data=output,
            file_name="sentiment_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
