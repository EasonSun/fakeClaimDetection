import numpy as np
import os
import re
import sys
import time
from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor
from lgExtractor import lgExtractor

experimentPath = sys.argv[1]
claimPath = experimentPath + 'claims.txt'
credPath = experimentPath + 'cred.npy'
logPath = experimentPath + 'log.txt'
sourcePath = experimentPath + 'source.npy'

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

lgPath = sys.argv[5]
articlePath = sys.argv[6]

'''
sourceCred
shape: numArticle, 4
value at each cell: float
oppose true | Support false | support true | oppose false 
'''
def evaluateSourceCred(source, stanceByArticle, cred):
	lc = len(source)
	sourceCred = np.zeros((len(source),4))
	for i in range(lc):	
		stance = stanceByArticle[i]
		cred_i = cred[i]
		q = stance&cred_i
		s = 2*q+cred_i
		# change this to numpy
		all_index = [idx for idx in range(len(source)) if source[idx] == source[i]]
		sourceCred[all_index, s] = sourceCred[all_index, s] + 1
	sourceCred4Training = np.zeros((len(source), 1))
	for i in range(lc):
		stance = stanceByArticle[i]
		cred_i = cred[i]
		q = stance&cred_i
		s = 2*q+cred_i
		sourceCred4Training[i] = sourceCred[i,s]

	# print(sourceCred)
	return sourceCred, sourceCred4Training


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
	credOrig = np.load(credPath)

	'''
	read articles and source 
	'''

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
	#claimSnippetIdx = []
	articleSnippetIdx = []
	relatedSnippets = []
	claimArticleIdx = []
	relatedArticles = []
	relatedSources= []
	cred = []	

	RSExtractor = relatedSnippetsExtractor(overlapThreshold)
	LGExtractor = lgExtractor(lgFeatures)

	curClaimIdx = 0
	curArticleIdx = 0
	for claim in claims:
		articles_ = articles[curArticleIdx]
		sources_ = sources[curArticleIdx]
		lgX_ = np.zeros(len(lgFeatures))
		for article, source in zip(articles_, sources_):
			relatedSnippets_, _ = RSExtractor.extract(claim, article)
			if relatedSnippets_ is not None:
				numRelatedSnippets_ = len(relatedSnippets_)
				# extract grature要等到最后
				# 要存一个relatedArticles
				articleSnippetIdx.extend([curArticleIdx for i in range(numRelatedSnippets_)])
				relatedSnippets.extend(relatedSnippets_)

				claimArticleIdx.append(curClaimIdx)
				relatedArticles.append(article)

				relatedSources.append(source)
				cred.append[credOrig[curClaimIdx]]
			curArticleIdx += 1
		curClaimIdx += 1

	'''
	relateRatio = len(claimSnippetIdx) / len(claims)
	print (relateRatio)
	logFile.write(relateRatio + '\n')
	'''
	numSnippet = len(articleSnippetIdx)
	assert(numSnippet == len(relatedSnippets))
	numArticle = np.unique(np.array(articleSnippetIdx)).shape
	assert (numArticle == len(relatedArticles))

	relatedSnippetsX, _, _ = RSExtractor.extractFeatures(relatedSnippets)
	stanceClf = Classifier(relatedSnippetX, 'stance', logPath, experimentPath)
	stanceProb = stanceClf.predict_porb()

	numClass = stanceProb.shape[1]
	stanceProbByArticle = np.zeros(numClass, numArticle)
	_, idx, counts = np.unique(articleSnippetIdx, return_inverse=True, return_counts=True)
	for i in range(numClass):
		stanceProbByArticle[i,:] = np.bincount(idx, weights=stanceProb[:,i]) / counts

	lgX = LGExtractor.extract(relatedArticles)
	assert(numArticle == lgX.shape[0])
	'''
	numFeature = lgX.shape[1]
	lgXByClaim = np.zeros(numFeature, numClaim)
	_, idx, counts = np.unique(claimArticleIdx, return_inverse=True, return_counts=True)
	for i in range(numFeature):
		lgXByClaim[i,:] = np.bincount(idx, weights=lgX[:,i]) / counts
	'''
	X = np.append(stanceProbByArticle.T, lgX, axis=1)
	assert(X.shape[1] == numFeature + numClass)
	y = np.array(cred)
	assert(X.shape[0] == y.shape)

	credClf = Classifier(X, 'cred', logPath, experimentPath, y)

	#sourceCred: [n_samples], or [numArticle]
	credClf.paramSearch()

	stanceByArticle = np.argmax(stanceProbByArticle, axis=0)
	sourceCredByStance, _ = evaluateSourceCred(relatedSources, stanceByArticle, cred)

	# (cv, numClaim, numClass)
	credClf.crossValidate(sourceCredByStance)

 	credClf.crossValidate(sourceCredByStance, claimArticleIdx)

	logFile.close()








