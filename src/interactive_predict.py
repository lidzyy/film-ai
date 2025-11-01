"""
1. Carregar modelo e vectorizer (dos arquivos .pkl)
2. Loop infinito:
   a) Pedir input do usuário
   b) Transformar texto em números
   c) Fazer previsão
   d) Mostrar resultado
   e) Perguntar: "Quer testar outro? (s/n)"
   f) Se "n" → sair do loop
"""

import joblib

model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
print("✅ Modelo carregado!")


print("🎬 Film AI - Análise de Sentimento")
print("=" * 50)

while True:
  review = input()
  review_vec = vectorizer.transform([review])
  prediction = model.predict(review_vec)[0]
  print(f"\n🎯 Sentimento: {prediction.upper()}")

  continuar = input('\nTestar outro review? s/n:')
  if continuar.lower() == "n":
    break

