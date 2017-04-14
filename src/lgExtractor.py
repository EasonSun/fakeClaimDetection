import numpy as np
import os

class lgExtractor(object):
    def __init__(self, lgFeatures):
        # to be used as the vocab of vectorizer
        self.lgFeatures = lgFeatures

    def extract(self, relatedArticles, numFeatures=3000):
        X = np.array(0)
        from sklearn.feature_extraction.text import TfidfVectorizer
        # empty string can be taken as all 0 vectors
        # using both uni- and bi-grams
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