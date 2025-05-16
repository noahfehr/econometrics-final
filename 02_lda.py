import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from gensim.models import Phrases
from gensim.models.phrases import Phraser
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('oecd-ai-all-ai-policies.csv')

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        bigram = Phrases(tokens, min_count=5, threshold=10)
        bigram_mod = Phraser(bigram)
        tokens_with_bigrams = bigram_mod[tokens]
        return ' '.join(tokens_with_bigrams)
    return ''

print("Preprocessing descriptions...")
processed_descriptions = df['Description'].drop_duplicates().apply(preprocess_text)

print("Creating document-term matrix...")
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
doc_term_matrix = vectorizer.fit_transform(processed_descriptions)

print("Applying LDA...")
n_topics = 2
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=200,
    learning_method='online',
    random_state=42,
    batch_size=128,
    verbose=0
)

lda_output = lda.fit_transform(doc_term_matrix)

feature_names = vectorizer.get_feature_names_out()

print("\nTop words in each topic:")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[:-10-1:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"\nTopic {topic_idx + 1}:")
    print(", ".join(top_words))

topic_columns = [f'Topic_{i+1}' for i in range(n_topics)]
df_topics = pd.DataFrame(lda_output, columns=topic_columns)
df = pd.concat([df, df_topics], axis=1)

print("\nExample documents for each topic:")
for topic_idx in range(n_topics):
    print(f"\nTop documents for Topic {topic_idx + 1}:")
    top_docs = df.nlargest(3, f'Topic_{topic_idx+1}')
    for idx, row in top_docs.iterrows():
        print(f"\nDescription: {row['Description'][:200]}...")
        print(f"Topic {topic_idx + 1} probability: {row[f'Topic_{topic_idx+1}']:.3f}")
