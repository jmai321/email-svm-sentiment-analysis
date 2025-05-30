# Spam Email Classification and Sentiment Analysis

This project is a Python-based implementation of spam email classification using Support Vector Machines (SVM) and sentiment analysis of spam emails. The project also includes parameter tuning using Grid Search and visualizations of results.

## Table of Contents
- [Features](#features)
- [Setup](#setup)
- [Dependencies](#dependencies)
- [Note](#note)

---

## Features

1. **Spam Classification**: 
   - Uses an SVM model to classify emails as spam or ham.
   - Includes parameter tuning using `GridSearchCV` for optimal performance.

2. **Sentiment Analysis**:
   - Analyzes spam emails for sentiment polarity using TextBlob.
   - Categorizes sentiments into Positive, Negative, and Neutral.

3. **Visualizations**:
   - Heatmap of accuracy scores for various SVM parameters.

     ![Grid Search Accuracy Heatmap](graph.png)

   - Bar graph showing the distribution of sentiment in spam emails.

     ![Sentiment Distribution Graph](sentiment%20graph.png)

---

## Setup

Follow these steps to set up the project environment:

1. Clone this repository:
   ```bash
   git clone https://github.com/jmai321/481-email-svm-sentiment-analysis.git
   cd 481-email-svm-sentiment-analysis
2. Create a virtual environment
   ```bash
   source venv/bin/activate # Linux/Mac
   .\venv\Scripts\activate # Windows
3. Install dependencies
   ```bash
   pip install pandas matplotlib scikit-learn textblob
4. Run the program
   ```bash
   python3 main.py

---
## Note

In order to view the second graph (the sentiment graph), close the previous graph (heatmap for SVM parameters)

