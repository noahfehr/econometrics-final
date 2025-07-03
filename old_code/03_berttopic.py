import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from gensim.models import Phrases
from gensim.models.phrases import Phraser
nltk.download('punkt')
nltk.download('stopwords')
df = pd.read_csv('oecd-ai-all-ai-policies.csv')

# Text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        bigram = Phrases(tokens, min_count=5, threshold=10) # TODO decide if we want to adjust this
        bigram_mod = Phraser(bigram)
        tokens_with_bigrams = bigram_mod[tokens]
        return ' '.join(tokens_with_bigrams)
    return ''

# Preprocess descriptions
print("Preprocessing descriptions...")
processed_descriptions = df['Description'].drop_duplicates().apply(preprocess_text)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', prediction_data=True)
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    min_topic_size=10,     # Merges tiny topics into larger ones, when set to 20 only 2 topics
    verbose=True
)
topics, probs = topic_model.fit_transform(processed_descriptions)

# Print the topics and their most representative terms
print("\nTop words for each topic:")
for topic_id in topic_model.get_topics():
    if topic_id != -1:  # Skip the outlier topic (-1)
        words = topic_model.get_topic(topic_id)
        print(f"\nTopic {topic_id}:")
        print(", ".join([word for word, _ in words[:10]]))  # Show top 10 words

print("\nExample documents per topic:")
for topic_id in topic_model.get_topics():
    if topic_id != -1:  # Skip the outlier topic (-1)
        docs = topic_model.get_representative_docs(topic_id)
        print(f"\nTopic {topic_id} example documents:")
        for doc in docs[:2]:  # Show 2 example documents per topic
            print(f"- {doc[:200]}...")  # Truncate long documents

