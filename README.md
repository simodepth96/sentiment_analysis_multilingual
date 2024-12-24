# Multilingual Sentiment Analysis on Product Reviews

This is a [bert-base-multilingual-uncased](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) model finetuned for sentiment analysis on product reviews in six languages: 
- English
- Dutch
- German
- French
- Spanish
- Italian

It predicts the sentiment of the review as a number of stars (between 1 and 5).

This model is intended for direct use as a sentiment analysis model for product reviews in any of the six languages above or for further finetuning on related sentiment analysis tasks.

## Requirements

For the purpose of this project, I've used the [Amazon Review Scraper Extension](https://chromewebstore.google.com/detail/amazon-review-scraper-exp/fmdnlcggjjlfimafoocabeghllidbopg) on a review page.

Here's an example of the input file you should upload on the app

![App Screenshot](amazon_scraped_reviews.png)

## Key Advantages of this Model for Sentiment Analysis

1. **Accuracy of the Sentiment's Magnitude**.
Compared to other models, such as [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), which return textual classifications (negative, neutral, positive), the **multilingual-uncased model** provides scores ranging from 1 to 5 to indicate the sentiment's magnitude. This approach helps disambiguate text classification for improved accuracy in the output (e.g., does a neutral review correspond to a score of 2 or 3?).

3. **Fine-tuned to Multilanguage text**. This model is able to capture the linguistic nuances from different langauges. Accuracy could be improved but still it does the job.



