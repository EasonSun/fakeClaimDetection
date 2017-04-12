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
lgPath = "data/linguisticFeatures/allFeatures.txt"

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

def main():
	logFile = open(logPath, 'a')

	claims = []
	with open(claimPath) as f:
		claims = f.readlines()
	lgFeatures = {}
	nextValue = 0
	with open(lgPath) as f:
		for lgFeature in f.readlines():
			lgFeatures[lgFeature] = nextValue
			nextValue += 1

	# idx to the orginal claim ids 
	# only contain the ones with a related reporting article
	# repeated for each snippets from the claim's reporting article
	# claim 	article 	snippet
	# 0			0			1
	# 0			0			2
	# 0			1			1
	# 0			6			4
	# 0			6			5
	##############################
	# claim 	article 
	# 0			0
	# 0			1
	# 0			6
	claimSnippetIdx = []
	claimArticleIdx = []
	curClaimIdx = 0
	relatedSnippets = []
	relatedArticles = []

	RSExtractor = relatedSnippetsExtractor(overlapThreshold)
	LGExtractor = lgExtractor(lgFeatures)

	for claim in claims:
		articles = articleCrawl(claim)
		lgX_ = np.zeros(len(lgFeatures))
		for article in articles:
			relatedSnippets_, _ = RSExtractor.extract(claim, article)
			if relatedSnippets_ is not None:
				numRelatedSnippets_ = len(relatedSnippets_)
				# extract grature要等到最后
				# 要存一个relatedArticles
				
				claimSnippetIdx.extend([curClaimIdx for i in range(numRelatedSnippets_)])
				relatedSnippets.extend(relatedSnippets_)

				claimArticleIdx.append(curClaimIdx)
				relatedArticles.append(article)
		curClaimIdx += 1

	'''
	relateRatio = len(claimSnippetIdx) / len(claims)
	print (relateRatio)
	logFile.write(relateRatio + '\n')
	'''

	relatedSnippetsX, _, _ = RSExtractor.extractFeatures(relatedSnippets)
	assert(len(claimSnippetIdx) == len(relatedSnippets))
	stanceClf = Classifier(relatedSnippetX, 'stance', logPath, experimentPath)
	stanceProb = stanceClf.predict_porb()
	assert(relatedSnippetsX.size == stanceProb.size)
	_, idx, counts = np.unique(claimSnippetIdx, return_inverse=True, return_counts=True)
	stanceProbByClaim = np.bincount(idx, weights=stanceProb.tolist()) / counts
	# is this one above supportiing?
	# If yes should 1 - it gets refuting?
	assert(stanceProbByClaim.size == len(claims))

	#问题是lgX每一个cell是一个array 
	lgX = LGExtractor.extract(relatedArticles).tolist()
	_, idx, counts = np.unique(claimArticleIdx, return_inverse=True, return_counts=True)
	lgXByClaim = np.bincount(idx, weights=lgX.tolist()) / counts
	

