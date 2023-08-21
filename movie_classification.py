# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 2023

@author: Aditya King
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Loading dataset 
print("Loading dataset...")
start_time = time.time()
data = pd.read_csv('H:\CODESOFT\Movie Genre Classification\movie_description\wiki_movie_plots_deduped.csv')
end_time = time.time()
print("Dataset loaded in {:.2f} seconds".format(end_time - start_time))

# Subset of the data for training and testing
subset_size = 10000  # Can change Subset size
subset_data = data.sample(n=subset_size, random_state=42)

# Splitting subset into features (plot summaries) and labels (genres)
X = subset_data['Plot']
y = subset_data['Genre']

# Splitting subset data into training and testing sets
print("Splitting data into train and test sets...")
start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
end_time = time.time()
print("Data split in {:.2f} seconds".format(end_time - start_time))

# Creating TF-IDF vectorizer
print("Creating TF-IDF vectorizer...")
start_time = time.time()
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # can change max_features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
end_time = time.time()
print("TF-IDF vectorization done in {:.2f} seconds".format(end_time - start_time))

# Creating and training Logistic Regression model
print("Training Logistic Regression model...")
start_time = time.time()
logreg_model = LogisticRegression(max_iter=1000, n_jobs=-1)  # Can change parameters
logreg_model.fit(X_train_tfidf, y_train)
end_time = time.time()
print("Logistic Regression training done in {:.2f} seconds".format(end_time - start_time))

# Predicting genre of movies in the testing set
print("Predicting genres...")
start_time = time.time()
y_pred = logreg_model.predict(X_test_tfidf)
end_time = time.time()
print("Prediction done in {:.2f} seconds".format(end_time - start_time))

# Finally Evaluating model
print("Evaluating model...")
start_time = time.time()
report = classification_report(y_test, y_pred, zero_division=1)

print(report)
end_time = time.time()
print("Evaluation done in {:.2f} seconds".format(end_time - start_time))
