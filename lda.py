import pandas as pd
import numpy as np
from pprint import pprint

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import string
import os

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import coherencemodel
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.models.wrappers import ldamallet

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from operator import itemgetter


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

os.environ.update({'MALLET_HOME': r'C:/new_mallet/mallet-2.0.8/'})
mallet_path = 'C:/new_mallet/mallet-2.0.8/bin/mallet'


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def compute_coherence_values(dnary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for topics in range(start, limit, step):
        model = LdaMallet(mallet_path, corpus=corpus, id2word=dnary, num_topics=topics, workers=3)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dnary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values


def get_review_topic(doc, dicty, model):
    doc = str(doc)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    bow = dicty.doc2bow(normalized.split())
    t = model.get_document_topics(bow)
    topic = max(t, key=itemgetter(1))
    return str(topic[0])


def visualize_topics(model, doc_term_matrix, dictionary):
    vis = pyLDAvis.gensim.prepare(model, doc_term_matrix, dictionary)
    return vis


def model_lda(clean_doc, dictionary, doc_term_matrix):

    lda_model = gensim.models.ldamodel.LdaModel(corpus=doc_term_matrix, id2word=dictionary, num_topics=25,
                                                random_state=100, update_every=1, chunksize=100, passes=25,
                                                alpha='auto', per_word_topics=True)
    print("Topics generated with the in-built LDA model are:\n")
    pprint(lda_model.print_topics())
    print("----------------------------------------------------")

    coherence_model_lda = CoherenceModel(model=lda_model, texts=clean_doc, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"coherence score: {coherence_lda}")

    return lda_model


def model_mallet(clean_doc, dictionary, doc_term_matrix):

    lda_mallet = LdaMallet(mallet_path, corpus=doc_term_matrix, id2word=dictionary, num_topics=25, workers=3)
    print("Topics generated with the mallet LDA model are:\n")
    pprint(lda_mallet.show_topics(formatted=False))
    print("----------------------------------------------------")

    coherence_model_mallet = CoherenceModel(model=lda_mallet, texts=clean_doc, dictionary=dictionary, coherence='c_v')
    coherence_mallet = coherence_model_mallet.get_coherence()
    print(f"coherence score: {coherence_mallet}")

    mallet_2 = ldamallet.malletmodel2ldamodel(lda_mallet)

    return mallet_2


def get_optimum_topics(df, dictionary, doc_term_matrix, clean_doc, start, limit):

    list_models, list_coherence = compute_coherence_values(dnary=dictionary, corpus=doc_term_matrix, texts=clean_doc,
                                                           limit=30, start=2, step=1)

    limit = limit
    start = start
    x = range(start, limit)
    plt.plot(x, list_coherence)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score")
    plt.show()

    final_model = list_models[list_coherence.index(max(list_coherence))]
    final_model_use = ldamallet.malletmodel2ldamodel(final_model)

    df["topic"] = df["reviewContent"].apply(get_review_topic, args=(dictionary, final_model_use))


def text_prepare(df):
    text_append = []
    for review in df.reviewContent:
        text_append.append(str(review))

    clean_doc = [clean(doc).split() for doc in text_append]
    dictionary = corpora.Dictionary(clean_doc)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in clean_doc]

    return clean_doc, dictionary, doc_term_matrix


def main():
    full_text = pd.read_csv("reviews_text_features.csv")
    pd.set_option('display.max_columns', None)
    full_text.drop('Unnamed: 0', axis=1, inplace=True)
    clean_doc, dictionary, doc_term_matrix = text_prepare(full_text)
    modellda = model_lda(clean_doc, dictionary, doc_term_matrix)
    vis_lda = visualize_topics(modellda, doc_term_matrix, dictionary)
    modelmallet = model_mallet(clean_doc, dictionary, doc_term_matrix)
    vis_mallet = visualize_topics(modelmallet, doc_term_matrix, dictionary)
    get_optimum_topics(full_text, dictionary, doc_term_matrix, clean_doc, start=2, limit=30)


if __name__ == "__main__":
    main()
