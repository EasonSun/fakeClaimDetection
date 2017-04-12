import numpy as np
import os

class LGExtractor(object):
	"""docstring for LGExtractor"""
	def __init__(self, lgFeatures):
		# to be used as the vocab of vectorizer
		self.lgFeatures = lgFeatures

	def extract(self, relatedArticles):
		X = np.array(0)
		from sklearn.feature_extraction.text import TfidfVectorizer
        # empty string can be taken as all 0 vectors
        # using both uni- and bi-grams
        vectorizer = TfidfVectorizer(analyzer = "word", \
                                     stop_words = "english",   \
		                             token_pattern = '(?u)\\b\\w\\w+\\b|!|\\?|\\"|\\\'', \
                                     #min_df=MIN_DF, \
                                     #max_df=MAX_DF, \
                                     ngram_range=(1, 2), \
                                     stop_words = None, \
                                     vocabulary=self.lgFeatures)    
        '''
        the min df above is really important as the first step for feature engineering
        .005 means only keep features apper more than .005 portion of docs
        that is roughly 486 docs
        '''
        lgX = vectorizer.fit_transform(relatedSnippets)
        # print (vectorizer.vocabulary_)
        lgX = lgX.toarray()
        return lgX