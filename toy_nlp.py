#loading corpus -> python
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Load labels
# in labels.csv, 1 means distressed and 0 means not distressed
labels = pd.read_csv("labels.csv")

#print(labels.head())

# Load text files (loop through each filename and read the corresponding .txt file)

texts = []
for fname in labels["filename"]:
    with open(os.path.join("data", fname), "r") as f:
        texts.append(f.read())

labels["text"] = texts

# vectorize text

vectorizer = TfidfVectorizer(
    # use 1-word and 2-word tokens, ignore english stop words
    ngram_range=(1,2),   # try (3,7) later
    stop_words="english"
)

X = vectorizer.fit_transform(labels["text"])

# dependent variable; what you want model to predict; 1 = distressed, 0 = not distressed
y = labels["distressed"]

# train logistic regression model
# split data so model is tested on unseen samples + ensures both classes appear in both sets
# class is distressed vs. not distressed, set is train vs. test

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# model learns patterns like liquidity issues -> distressed
model = LogisticRegression()
model.fit(X_train, y_train)

# model outputs 0 or 1 for each test sample, based on what it learned from the training data

preds = model.predict(X_test)

# matrix
print(classification_report(y_test, preds))