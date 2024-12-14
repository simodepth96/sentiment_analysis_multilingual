import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Load the Excel file
data_path = "/content/Amazon Review Scraper.xlsx"
df_2 = pd.read_excel(data_path)

# Check if the 'review' column exists
if 'review' not in df_2.columns:
    raise ValueError("The uploaded file does not contain a 'Review' column.")

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

df_2['sentiment'] = df_2['review'].apply(analyze_sentiment)

#scatterplot to showcase the distribution of the sentiment magnitude
# Define a function to calculate and classify emotion
def calculate_emotion(smag):
    if smag > 0 and smag < 2:
        sent_m_label = "Poor"
    elif smag >= 5:
        sent_m_label = "Positive"
    elif smag > 3 and smag < 4: # changed from >=5 to >3 and <4
        sent_m_label = "Neutral"
    elif smag >= 2 and smag <=3: #added to address values not captured above
        sent_m_label = "Poor"
    elif smag >=4 and smag <5: #added to address values not captured above
        sent_m_label = "Neutral"
    else:
        sent_m_label = "Undefined Emotion"
    return sent_m_label

# Prepare data for the scatter plot
predictedY = []
emotion_labels = []
plotcolors = []

for sentiment_value in df_2['sentiment']:
    smag = sentiment_value  # Assuming sentiment_value is used as the magnitude
    predictedY.append(smag)
    
    emotion_label = calculate_emotion(smag)
    emotion_labels.append(emotion_label)
    
    # Set the color based on emotion level - Modified to cover all cases
    if smag > 0 and smag < 2:
        plotcolors.append('red')
    elif smag >= 5:
        plotcolors.append('green')
    elif smag > 3 and smag < 4: # changed from >=5 to >3 and <4
        plotcolors.append('yellow')
    elif smag >= 2 and smag <= 3: # color for Poor (2 and 3)
        plotcolors.append('red')  
    elif smag >= 4 and smag < 5: # color for Neutral (4)
        plotcolors.append('yellow')
    else:
        plotcolors.append('gray') # color for undefined emotion (just in case)

# Scatter plot of sentiment magnitudes with emotion labels
plt.figure(figsize=(14, 6))
plt.scatter(predictedY, np.zeros_like(predictedY), color=plotcolors, s=150)

# Adjustments to make the plot clearer
plt.yticks([])
plt.xlim(0, 5)
plt.xlabel('Poor                         Neutral                                 Positive')
plt.title("Sentiment Magnitude Analysis")

# Show the plot
plt.show()

# Save the results to a new Excel file
df_2.to_excel("sentiment_analysis_results.xlsx", index=False)
df_2.head()
