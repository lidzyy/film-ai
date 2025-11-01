import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/raw/movie_reviews.csv')
X = df['review']
y = df['sentiment']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

new_reviews = [
    "This movie was incredible, I loved every second",
    "Terrible film, complete waste of my time",
    "i waS there with my girlfriend",
]

for review in new_reviews:
    review_vec = vectorizer.transform([review])
    prediction = model.predict(review_vec)[0]
    print(f"\nReview: {review}")
    print(f"Previsão: {prediction.upper()}")