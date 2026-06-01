#%% import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% colums and row options
pd.set_option('display.max_rows', None)  # Satır sınırını kaldır
pd.set_option('display.max_colwidth', None)  # Sütundaki metin uzunluğu sınırını kaldır

#%% reading data
data = pd.read_csv(r'NLP/gender-classifier.csv', encoding="latin1")
data = pd.concat([data.gender, data.description], axis=1)
data.dropna(axis=0, inplace=True)
data.gender = [1 if each == 'female' else 0 for each in data.gender]

#%% cleaning data
import re
import nltk  # natural language toolkit
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
from nltk.corpus import stopwords

# Define a function for data cleaning
def clean_text(description):
    description = re.sub("[^a-zA-Z]", " ", description)  # Remove non-alphabetical characters
    description = description.lower()  # Convert to lowercase
    description = nltk.word_tokenize(description)  # Tokenize text
    #description = [word for word in description if word not in set(stopwords.words("english"))]  # Remove stopwords
    lemma = nltk.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]  # Lemmatize
    return " ".join(description)  # Join words back into a single string

# Apply cleaning to all descriptions
description_list = [clean_text(desc) for desc in data.description]

#%% Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
max_features = 600

count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

#%% Split data into train and test
y = data.iloc[:, 0].values  # Gender labels
x = sparce_matrix  # Bag of Words matrix

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

#%% Naive Bayes model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

#%% Prediction and accuracy
y_pred = nb.predict(x_test)
accuracy = nb.score(x_test, y_test)  # Use x_test and y_test to calculate accuracy
print("Accuracy:", accuracy)
