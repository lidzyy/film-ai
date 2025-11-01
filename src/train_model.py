import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/raw/movie_reviews.csv')

X = df['review'] # feature (input)
y = df['sentiment'] # labels (output)

X_train, X_test, y_train, y_test = train_test_split(
  X,
  y,
  test_size=0.2,
  random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_vec, y_train) # treinar
y_pred = model.predict(X_test_vec) # Prever

print(y_pred)
"""
As previsoes estao certas?
Quantas acertou?
Quantas errou?
Taxa de acerto?

Proximo Passo:
  - avaliar o modelo
"""