import numpy as np
import os
import re
import sys
import time
from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor

experimentPath = sys.argv[1]
claimPath = experimentPath + 'claims.txt'
credPath = experimentPath + 'cred.npy'
logPath = experimentPath + 'log.txt'

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

def main():
	claims = []
	with open(claimPath) as f:
		claims = f.readlines()
	claims = [x.strip() for x in claims]
	relatedSnippets = []
	relatedSnippetsX = []
	claimIdx = []
	curIdx = 0
	numRelatedArticles = 0	# to log

	for claim in claims:
		articles = articleCrawl(claim)
		for article in articles:
			relatedSnippets_, _ = extractor.extract(claim, article)
			numRelatedSnippets_ = len(relatedSnippets_)
			if relatedSnippets_ is not None:
				numRelatedArticles += 1
				relatedSnippetsX_, _, _ = extractor.extractFeatures(relatedSnippets_)
				
				claimIdx.extend([curIdx for i in range(numRelatedSnippets_)])
				relatedSnippets.extend(relatedSnippets_)
				relatedSnippetsX.extend(relatedSnippetsX_.tolist())
		curIdx += 1
	assert(len(claimIdx) == len(relatedSnippets))
	stanceClf = Classifier(relatedSnippetX, 'stance', logPath, experimentPath)
	stanceProb = stanceClf.predict_porb()
	assert(relatedSnippetsX.size == stanceProb.size)
	_, idx, counts = np.unique(claimIdx, return_inverse=True, return_counts=True)
	stanceProbByClaim = np.bincount(idx, weights=stanceProb.tolist()) / counts
	assert(stanceProbByClaim.size == curIdx)

	# lg features 
	




from sklearn.externals import joblib
estimatorPath = experimentPath + 'rf.pkl'
rf = joblib.load(estimatorPath)

