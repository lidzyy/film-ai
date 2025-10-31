import pandas as pd

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
all_positive_text = positive_texts.str.cat(sep=' ')
negative_texts = negative_df['review']
all_negative_text = negative_texts.str.cat(sep=' ')