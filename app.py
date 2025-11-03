from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# carregar modelos
model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
  review = request.json.get('review', '')

  if not review:
    return jsonify({'error' : 'Review vazio'})
  
  # Transformar e prever
  review_vec = vectorizer.transform([review])
  prediction = model.predict(review_vec)

  return jsonify({
      'sentiment': prediction[0],
      'review' : review

  })

if __name__ == "__main__":
  app.run(debug=True)