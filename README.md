# Financial Market News - Sentiment Analysis

## Overview
This project aims to predict the overall sentiment (positive or negative) of financial news articles. The dataset contains the top 25 news articles for each day, and the goal is to train a machine learning model to classify the sentiment based on the news content. The Random Forest Classifier is used for this task.

## Dataset
The dataset consists of financial news headlines collected daily, with each row containing the top 25 news articles for a specific day. The \`Label\` column represents the sentiment, where:
- **1** indicates a positive sentiment
- **0** indicates a negative sentiment

## Approach
1. The dataset is loaded and preprocessed by combining all news articles into a single text entry per row.
2. The text data is converted into a numerical format using a bag-of-words approach.
3. The dataset is split into training and testing sets.
4. A machine learning model, specifically a Random Forest Classifier, is trained on the dataset.
5. The model's performance is evaluated using a confusion matrix and a classification report.

## Model Performance
- The model achieves an accuracy of **50%**.
- The positive class (**1**) is predicted with **higher recall**, meaning the model identifies positive sentiments more effectively.
- The negative class (**0**) has **lower precision and recall**, indicating that further improvements are required.

## Future Enhancements
- Implement advanced feature extraction techniques such as TF-IDF.
- Explore deep learning models like LSTMs or Transformers for better text analysis.
- Optimize hyperparameters of the machine learning model for improved accuracy.

## Requirements
To run this project, install the required dependencies:
- pandas
- numpy
- scikit-learn
- matplotlib (optional for visualization)




