# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:22:28 2017

@author: sin-ev
"""

from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
import logging
import numpy as np
import re

def overlap(snippets,claim,num_topics=2):
    # print(snippets) if (snippets == [])
    #the snippets of one article, and one claim from the same article
    #num_topics is the number of topics that we want to classify
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    #texts = [[word for word in document.lower().split()] for document in snippets]
    claim = " ".join(re.findall("[a-zA-Z0-9]+", claim))
    texts = []
    # print(type(claim))
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english', \
                             ngram_range=(1, 2))        
    for i in range(len(snippets)):
        try:
            vectorizer.fit_transform([snippets[i]])
        except ValueError:
            print (claim)
            print (snippets)
            return
        texts.append(list(vectorizer.vocabulary_))

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    # print(tfidf)
    corpus_tfidf = tfidf[corpus]
    # print(corpus_tfidf)
    #lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    try:
        lsi = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    except (ValueError, RuntimeWarning):
        print (claim)
        print (snippets)
    corpus_lsi = lsi[corpus_tfidf]
    # print(corpus_lsi)
    index = similarities.MatrixSimilarity(lsi[corpus])
    # print (index)
    # print (claim.lower().split())
    claim_bow = dictionary.doc2bow(claim.lower().split())
    # print(claim_bow)
    claim_lsi = lsi[claim_bow]
    # print (claim_lsi)
    sims = np.array(index[claim_lsi])
    return corpus_lsi, sims