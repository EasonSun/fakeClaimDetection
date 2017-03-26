# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 07:22:28 2017

@author: sin-ev
"""

from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
import logging

def overlap(snippets,claim,num_topics=2):
    #the snippets of one article, and one claim from the same article
    #num_topics is the number of topics that we want to classify
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    texts = [[word for word in document.lower().split()] for document in snippets]
    """    
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             ngram_range=(1, 2))        
    featureVector = vectorizer.fit_transform(snippets)
    """
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    corpus_lsi = lsi[corpus_tfidf]
    index = similarities.MatrixSimilarity(lsi[corpus])
    claim_bow = dictionary.doc2bow(claim.lower().split())
    claim_lsi = lsi[claim_bow]
    sims = index[claim_lsi]
    return corpus_lsi,sims