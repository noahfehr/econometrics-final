import pandas as pd
import numpy as np
import nltk
from nltk import sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from random import shuffle
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
import re
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

nltk.download('punkt')
nltk.download('stopwords')

translator = str.maketrans('','',punctuation) 
stemmer = SnowballStemmer('english')
stoplist = set(stopwords.words('english'))


def add_full_text(agora_id):
    """
    Append to a new column the full text for each bill stored in the fulltext 
    folder in the agora dataset.

    Keyword arguments:
    agora_id - id for each artifact in the agora dataset
    """
    text = None
    try:
        with open(f'agora/fulltext/{agora_id}.txt', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError as e:
        print(e)
    return text


def preprocess_text(text):
    """
    Preprocess the full text of each bill text for the BERT model. This includes 
    setting all text to lowercase, tokenizing, removing stopwords and all legalese, 
    and creating bigrams.

    Keyword arguemnts:
    text - full text for each law in the agora dataset
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in list(stop_words) + common_law_terms]
        bigram = Phrases(tokens, min_count=5, threshold=10)
        bigram_mod = Phraser(bigram)
        tokens_with_bigrams = bigram_mod[tokens]
        return ' '.join(tokens_with_bigrams)
    return ''


def normalize_text(text):
    """
    Preprocess the full text of each bill text for the word2vec word embedding model. 
    This includes setting all text to lowercase, removing punctuation, removing stop words, 
    replacing numbers with #, and stemming words. 

    Keyword arguemnts:
    text - full text for each law in the agora dataset
    """
    text = text.replace('\r', ' ').replace('\n', ' ')
    lower = text.lower() # all lower case
    nopunc = lower.translate(translator) # remove punctuation
    words = nopunc.split() # split into tokens
    nostop = [w for w in words if w not in stoplist] # remove stopwords
    no_numbers = [w if not w.isdigit() else '#' for w in nostop] # normalize numbers
    stemmed = [stemmer.stem(w) for w in no_numbers] # stem each word
    return no_numbers


def get_sentences(text):
    """
    Splits each bill text by sentence and normalizes the sentence text. 

    Keyword argument:
    text - full text for each law in the agora dataset
    """
    sent=[]
    for raw in sent_tokenize(text):
        raw2 = normalize_text(raw)
        sent.append(raw2)
    return sent

if __name__ == '__main__':
    df = pd.read_csv('agora/documents.csv')

    # add the full bill text to the dataframe via a column called full_text
    df["full_text"] = df["AGORA ID"].apply(add_full_text)
    # df.to_csv('agora_raw.csv')

    sample = list(df["full_text"])

    # get a list of all sentences in each bill text 
    sentences = []
    for doc in sample:
        try:
            sentences += get_sentences(doc)
        except:
            pass

    shuffle(sentences)

    # train a word2vec model with the shuffled sentences; output is a 300-dimensional
    # vector for each word that appears a minimum of 25 times in the sentences corpus
    w2v = Word2Vec(sentences, workers = 8, vector_size=300, min_count =  25, 
                   window = 5, sample = 1e-3)
    
    # read all legal terms in the .txt file and convert into a list
    with open('legal_words.txt') as file:
        common_law_terms = file.read()
    common_law_terms = common_law_terms.split(',')

    # for each base law term, get 5 most similar terms in corpus by word embedding
    common_law_similar = list()
    for term in common_law_terms:
        try:
            for similar_term, _ in w2v.wv.most_similar(term)[:5]:
                common_law_similar.append(similar_term)
        except:
            pass

    # append both list to get a list of legalese terms
    common_law_terms = common_law_terms + common_law_similar

    df["full_text_preprocessed"] = df["full_text"].apply(preprocess_text)
    # df.to_csv("agora_processed.csv")

    # instantiate BERTopic model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    hdbscan_model = HDBSCAN(min_cluster_size=20, metric='euclidean', prediction_data=True)
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        min_topic_size=10,
        verbose=True,
        calculate_probabilities=True
    )

    topics, probs = topic_model.fit_transform(df['full_text_preprocessed'])

    print("\nTop words for each topic:")
    for topic_id in topic_model.get_topics():
        if topic_id != -1:  # Skip the outlier topic (-1)
            words = topic_model.get_topic(topic_id)
            print(f"\nTopic {topic_id}:")
            print(", ".join([word for word, _ in words[:10]]))  # Show top 10 words

    BERT_probs = pd.DataFrame(probs, columns=['BERT_topic0', 'BERT_topic1', 'BERT_topic2',
                                               'BERT_topic3', 'BERT_topic4', 'BERT_topic5', 
                                               'BERT_topic6', 'BERT_topic7'])
    
    df = pd.concat([df, BERT_probs], axis=1)

    # remove all agora artifacts which are part of the topic -1 (dummy topic)
    dummy_topic_indices = np.where(np.array(topics) == -1)[0]
    df = df.drop(dummy_topic_indices)
    df = df.reset_index(drop=True)
    # df.to_csv('agora_topic_probabilities.csv')
