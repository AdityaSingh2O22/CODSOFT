# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:03:10 2023

@author: Aditya King
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Loading of dataset 
data = pd.read_csv(r'H:\CODESOFT\Spam SMS Detection\spam_sms_detection_dataset\spam.csv', encoding='ISO-8859-1')

# Splitting data into X and y (target)
X = data['v2']
y = data['v1']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Can change parameter
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Building and training the model using - (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predicting
y_pred = model.predict(X_test_tfidf)

# Evaluating the model
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
