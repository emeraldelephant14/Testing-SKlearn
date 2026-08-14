#loading corpus -> python
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load labels
labels = pd.read_csv("labels.csv")

# Load text files
texts = []
for fname in labels["filename"]:
    with open(os.path.join("data", fname), "r") as f:
        texts.append(f.read())

labels["text"] = texts

#vectorize text

vectorizer = TfidfVectorizer(
    ngram_range=(1,2),   # try (3,7) later
    stop_words="english"
)

X = vectorizer.fit_transform(labels["text"])
y = labels["distressed"]