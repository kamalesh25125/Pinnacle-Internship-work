import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


print("=" * 50)
print("MOVIE REVIEW SENTIMENT ANALYSIS")
print("=" * 50)

# Load Dataset
print("\nLoading IMDb Dataset...")

df = pd.read_csv("IMDB Dataset.csv")

print(f"Dataset Loaded Successfully!")
print(f"Total Reviews: {len(df)}")

# Features and Labels
X = df["review"]
y = df["sentiment"]

# Convert Text to Numerical Features
print("\nConverting Reviews into TF-IDF Features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

# Train-Test Split
print("Splitting Dataset into Training and Testing Sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
print("Training Multinomial Naive Bayes Model...")

model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%")
print("=" * 50)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.show()

# User Prediction Loop
print("\nSentiment Prediction System Ready!")
print("Type 'exit' to quit.")

while True:

    review = input("\nEnter Review: ")

    if review.lower() == "exit":
        print("\nThank you for using the Sentiment Analysis System.")
        break

    if review.strip() == "":
        print("Please enter a valid review.")
        continue

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    if prediction == "positive":
        print("Sentiment: POSITIVE 😊")
    else:
        print("Sentiment: NEGATIVE 😞")