# Spam Mail Predictor

A machine learning web application that classifies emails as **Spam** or **Ham (Safe)** using Natural Language Processing (NLP).


## Overview

This project builds an end-to-end spam email classifier using text processing, feature engineering, and multiple ML models.
The web interface (built with **Gradio**) allows real-time predictions of whether an entered email is spam or not.


## Features

* Text cleaning (URL removal, stopword filtering, lemmatization)
* TF-IDF vectorization for feature extraction
* Multiple models trained and compared:
  * Logistic Regression
  * Naive Bayes *(used in the app)*
  * Support Vector Machine (Linear)
* Evaluation metrics: Accuracy, Classification Report, Confusion Matrix
* WordCloud visualizations for spam vs ham emails
* Deployed via Gradio for instant predictions


## Tech Stack used:

* **Python**
* **Scikit-learn**
* **NLTK**
* **Pandas, NumPy, Matplotlib, Seaborn**
* **Gradio** (for deployment)

## Model Performance

After comparison, **SVM (Linear)** achieved the highest accuracy and was selected for deployment.

## Sample Output

| Input                                        | Prediction |
| -------------------------------------------- | ---------- |
| “Congratulations! You’ve won a free iPhone!” | Spam    |
| “Let’s meet for coffee tomorrow.”            | Ham      |


## Scope for Improvements

* Add deep learning models (LSTM, BERT)
* Expand dataset for better generalization