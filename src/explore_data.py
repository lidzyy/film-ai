import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


df = pd.read_csv('data/raw/movie_reviews.csv')

print("=" * 50)
print("PRIMEIRAS 5 REVIEWS")
print("=" * 50)
print(df.head()) 

print("\n" + "=" * 50)
print("INFORMAÇÕES DO DATASET")
print("=" * 50)
df.info()

sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

positive_df = df[df['sentiment'] == 'positive']
negative_df = df[df['sentiment'] == 'negative']

positive_texts = positive_df['review']
all_positive_text = positive_texts.str.cat(sep=' ').lower()

for char in string.punctuation:
  all_positive_text = all_positive_text.replace(char, '')
all_positive_text = all_positive_text.split(' ')

negative_texts = negative_df['review']
all_negative_text = negative_texts.str.cat(sep=' ').lower()

for char in string.punctuation:
  all_negative_text = all_negative_text.replace(char, '')
all_negative_text = all_negative_text.split(' ')

positive_filtered = [word for word in all_positive_text if word not in stop_words]
negative_filtered = [word for word in all_negative_text if word not in stop_words]

positive_counter = Counter(positive_filtered)
negative_counter = Counter(negative_filtered)

print(positive_counter.most_common(5))
print(negative_counter.most_common(5))