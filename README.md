# 🎬 Film AI - Movie Sentiment Analysis

Projeto de Machine Learning para análise de sentimento de reviews de filmes usando Python e Scikit-learn.

## 📊 Resultados

- **Taxa de acerto:** 98.36%
- **Dataset:** 300 reviews balanceadas
- **Algoritmo:** Logistic Regression com TF-IDF (N-grams)

## 🚀 Funcionalidades

- ✅ Classificação binária (Positivo/Negativo)
- ✅ Suporte a N-grams (entende negações!)
- ✅ Modelo salvo e reutilizável
- ✅ Interface interativa para testes


## 🛠️ Tecnologias

- Python 3.x
- pandas
- scikit-learn
- NLTK
- joblib

## ⚙️ Instalação

```bash
# Clone o repositório
git clone https://github.com/lidzyy/film-ai.git
cd film-ai

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt

# Baixar stop words (primeira vez)
python -c "import nltk; nltk.download('stopwords')"

## Treinar Modelo
python src/train_model.py

## Modo Interativo
python src/interactive_predict.py

## 🧠 O Que o Modelo Aprendeu

**Entende:**
- ✅ Sentimentos positivos: "amazing", "loved", "best"
- ✅ Sentimentos negativos: "terrible", "worst", "awful"
- ✅ Negações: "don't like", "didn't enjoy", "not good"
- ✅ Expressões: "worst film ever", "waste of time"

## 📚 Aprendizagens

Durante o desenvolvimento deste projeto, aprendi:

- **Importância de dados equilibrados:** Conjuntos de dados desequilibrados levam a modelos tendenciosos
- **N-gramas para capturar contexto:** Bigramas (1,2) permitem compreender negações como "don't like"
- **Pré-processamento de texto:** Minúsculas, remoção de pontuação e palavras vazias melhoram a performance
- **Iteração e melhoria contínua:** De 16% para 98% através de experimentação sistemática

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se livre para abrir issues ou pull requests.

## 📝 Licença

MIT License
