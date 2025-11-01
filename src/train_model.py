import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

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
print(y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Taxa de acerto: {accuracy:.2%}")
print(classification_report(y_test, y_pred))

joblib.dump(model, 'models/sentiment_model.pkl')
print("✅ Modelo salvo: models/sentiment_model.pkl")

joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Vectorizer salvo: models/vectorizer.pkl")
print("\n🎉 Modelo pronto para uso!")
