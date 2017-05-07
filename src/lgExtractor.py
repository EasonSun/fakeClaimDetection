import numpy as np
import os

class lgExtractor(object):
    def __init__(self, lgPath):
        # to be used as the vocab of vectorizer
        lgFeatures = {}
        nextValue = 0
        with open(lgPath) as f:
            for lgFeature in f:
                lgFeature = lgFeature.rstrip()
                if lgFeature not in lgFeatures:
                    lgFeatures[lgFeature] = nextValue
                    nextValue += 1
        self.lgFeatures = lgFeatures

    def extract(self, relatedArticles, numFeatures=3000):
        X = np.array(0)
        from sklearn.feature_extraction.text import TfidfVectorizer
        # empty string can be taken as all 0 vectors
        # lgFeatures contain uni- and bi-grams
        # cannot use TF-IDF when vocab is given, instead use top numFeature of features.
        vectorizer = TfidfVectorizer(analyzer = "word", \
                                    token_pattern = '(?u)\\b\\w\\w+\\b|!|\\?|\\"|\\\'', \
                                    vocabulary=self.lgFeatures, \
                                    ngram_range=(1, 2))  
        '''
        the min df above is really important as the first step for feature engineering
        .005 means only keep features apper more than .005 portion of docs
        that is roughly 486 docs
        '''
        lgX_ = vectorizer.fit_transform(relatedArticles)
        topFreqIdx = vectorizer.idf_.argsort()[:numFeatures]
        #print (topFreqIdx.shape)
        lgX = lgX_[:, topFreqIdx]
        # print (vectorizer.vocabulary_)
        lgX = lgX.toarray()
        featureNames = vectorizer.get_feature_names()
        featureNames = np.array(featureNames)[topFreqIdx]
        return lgX, featureNames