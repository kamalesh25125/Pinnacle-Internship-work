import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("IMDB Dataset.csv")

X = df["review"]
y = df["sentiment"]


# ----------------------------
# TF-IDF Feature Extraction
# ----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)


# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ----------------------------
# Train Model
# ----------------------------
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)


# ----------------------------
# Calculate Accuracy
# ----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analysis")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.write(
    "Analyze whether a movie review is Positive or Negative using Machine Learning."
)

st.divider()

st.write("### Project Information")

st.write("**Dataset:** IMDb 50K Movie Reviews")
st.write("**Algorithm:** Logistic Regression")
st.write("**Feature Extraction:** TF-IDF (10,000 Features)")
st.write("**N-Grams:** Unigrams + Bigrams")

st.divider()

review = st.text_area(
    "Enter a movie review:",
    height=150,
    placeholder="Example: This movie was fantastic and I loved every moment..."
)

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        if prediction == "positive":

            st.success("😊 Positive Review")

            st.write(
                "The model predicts that the review expresses a positive sentiment."
            )

        else:

            st.error("😞 Negative Review")

            st.write(
                "The model predicts that the review expresses a negative sentiment."
            )