import numpy as np
import os
import gc
import re
import sys
sys.path.append('src/') # This line is added because otherwise it will show ModuleNotFound error
import time
import json
import psutil
import _pickle as pickle
from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor
from lgExtractor import lgExtractor

experimentPath = sys.argv[1]
logPath = experimentPath + 'log.txt'
sourcePath = experimentPath + 'source.npy'

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

lgPath = sys.argv[5]
snopeDataPath = sys.argv[6]
googleDataPath = sys.argv[7]

def readSnopes(filePath):
	filePath = os.path.join(snopeDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	if data['Credibility'] in ['true', 'mostly true']:
		return data['Claim'], 0# for
	elif data['Credibility'] in ['false', 'mostly false']: 
		return data['Claim'], 1


def readGoogle(filePath):
	filePath = os.path.join(googleDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	return data['article'], data['source']


'''
sourceCred
shape: numArticle, 4
value at each cell: float
oppose true | Support false | support true | oppose false 
'''
def evaluateSourceCred(sources, stanceByArticle, cred):
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
	#logFile = open(logPath, 'a')
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
	creds = []	

	RSExtractor = relatedSnippetsExtractor(overlapThreshold)
	LGExtractor = lgExtractor(lgPath)

	curClaimIdx = 0
	curArticleIdx = 0
	everythingPath = os.path.join(experimentPath, 'everything')
	numClaimsInGoogleResults = 3576 
	groupSize = 149
	print("groupSize: "+str(groupSize))
	proc = psutil.Process(os.getpid())

	if not os.path.isfile(everythingPath):
		print ('reading data')
		_numClaim = 0
		_numArticle = 0
		# each is a claim
		for filePath in os.listdir(googleDataPath):
			_numClaim += 1

			if not filePath.endswith('.json'):
				continue
			print('Enter 1st loop')
			articles_, sources_ = readGoogle(filePath)
			claim, cred = readSnopes(filePath)
			
			print("Before 2nd loop:")
			for article, source in zip(articles_, sources_):

				_numArticle += 1
				if (_numArticle == 19):
					print ('darn')
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
					creds.append(cred)
				curArticleIdx += 1
				print("after 1 article:")
			curClaimIdx += 1

			
			# 
			if _numClaim % groupSize == 0:
				
				f = open(everythingPath+str(_numClaim), 'wb')
				pickle.dump(articleSnippetIdx, f)
				pickle.dump(relatedSnippets, f)
				pickle.dump(claimArticleIdx, f)
				pickle.dump(relatedArticles, f)
				pickle.dump(relatedSources, f)
				pickle.dump(creds, f)

				del articleSnippetIdx
				del relatedSnippets
				del claimArticleIdx
				del relatedArticles
				del relatedSources
				del creds
				gc.collect()

				articleSnippetIdx = []
				relatedSnippets = []
				claimArticleIdx = []
				relatedArticles = []
				relatedSources= []
				creds = []
				
				print (_numClaim, _numArticle)
	else:
		print ('loading data')
		f = open(everythingPath, 'rb')
		articleSnippetIdx = pickle.load(f)
		relatedSnippets = pickle.load(f)
		claimArticleIdx = pickle.load(f)
		relatedArticles = pickle.load(f)
		relatedSources= pickle.load(f)
		creds = pickle.load(f)	

	return

	'''
	relateRatio = len(claimSnippetIdx) / len(claims)
	print (relateRatio)
	logFile.write(relateRatio + '\n')
	'''
	print ('stance feature')
	numSnippet = len(articleSnippetIdx)
	assert(numSnippet == len(relatedSnippets))
	numArticle = np.unique(np.array(articleSnippetIdx)).shape[0]
	assert (numArticle == len(relatedArticles))

	relatedSnippetX, _ = LGExtractor.extract(relatedSnippets, numFeatures=2000)
	stanceClf = Classifier(relatedSnippetX, 'stance', logPath, experimentPath)
	# use trained?
	stanceProb = stanceClf.predict_porb()

	numClass = stanceProb.shape[1]
	stanceProbByArticle = np.zeros(numClass, numArticle)
	_, idx, counts = np.unique(articleSnippetIdx, return_inverse=True, return_counts=True)
	for i in range(numClass):
		stanceProbByArticle[i,:] = np.bincount(idx, weights=stanceProb[:,i]) / counts

	print ('lg feature')
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
	y = np.array(creds)
	assert(X.shape[0] == y.shape)

	print ('classification')
	credClf = Classifier(X, 'cred', logPath, experimentPath, y)

	#sourceCred: [n_samples], or [numArticle]
	credClf.paramSearch()

	stanceByArticle = np.argmax(stanceProbByArticle, axis=0)
	sourceCredByStance, _ = evaluateSourceCred(relatedSources, stanceByArticle, cred)

	# (cv, numClaim, numClass)
	# per article
	credClf.crossValidate(sourceCredByStance)

	# per claim
	credClf.crossValidate(sourceCredByStance, claimArticleIdx)

	logFile.close()

if __name__ == '__main__':
    main()







