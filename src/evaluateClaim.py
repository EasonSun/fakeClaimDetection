import numpy as np
import os
import sys
import time
import json
import io
import heapq
from multiprocessing import Pool, Manager

import pickle
from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor
from lgExtractor import lgExtractor
from ClaimReader import ClaimReader


experimentPath = sys.argv[1]
logPath = experimentPath + 'log.txt'
sourcePath = experimentPath + 'sourceMatrix_' + sys.argv[9] + '_' + sys.argv[10]
reviewPath = experimentPath + 'reviews_' + sys.argv[9] + '_' + sys.argv[10]

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

lgPath = sys.argv[5]
snopeDataPath = sys.argv[6]
googleDataPath = sys.argv[7]
doc2vecPath = sys.argv[8]


filePaths = os.listdir(googleDataPath)
global reviews
global sourceMatrix
'''
manager = Manager()
reviews = manager.list()
sourceMatrix = manager.dict()
'''
reviews = list()
sourceMatrix = dict()

reader = ClaimReader(snopeDataPath, googleDataPath)
rsExtractor = relatedSnippetsExtractor(overlapThreshold, doc2vecPath=doc2vecPath)
stanceClf = Classifier('stance', logPath, experimentPath)


class topK(object):
	# @param {int} k an integer
	def __init__(self, k):
		self.k = k
		self.nums = []
		heapq.heapify(self.nums)
		self.newAdd = None

	# @param {int} num an integer
	def add(self, nums):
		nums = list(nums)
		for num in nums:
			if len(self.nums) < self.k:
				heapq.heappush(self.nums, num)
			elif num > self.nums[0]:
				heapq.heappop(self.nums)
				heapq.heappush(self.nums, num)
		self.newAdd = set(self.nums).intersection(nums)

	# @return {int[]} the top k largest numbers in array
	def topk(self):
		return sorted(self.nums, reverse=True)

	def avg(self):
		return sum(self.nums) / len(self.nums)


class Review(object):
	"""docstring for ClassName"""
	def __init__(self, claim, label):
		# number of claims
		self.claim = claim
		self.label = label

	def addSnippets(self,  posTKrelatedSnippets, negTKrelatedSnippets):
		# top 10
		self.posTKrelatedSnippets = posTKrelatedSnippets
		self.negTKrelatedSnippets = negTKrelatedSnippets
		#self.snippetsScore = snippetsScore

	def addArticles(self, articlesScore, source):
		# number of articles
		self.articlesScore = articlesScore
		self.source = source

# sourceMatrix
# {source: (#support true, #support false, #refute true, #refute false)}
def updateSource(articlesScore, source, cred):
	if source not in sourceMatrix:
		# smoothing
		sourceMatrix[source] = [1,1,1,1]
	#print (cred)
	#print (articlesScore)
	# support
	if articlesScore[0]*2.4 > articlesScore[1]:
		# true
		if cred == 0:
			sourceMatrix[source][0] += 1
		else:
			sourceMatrix[source][1] += 1
	else:
		if cred == 0:
			sourceMatrix[source][2] += 1
		else:
			sourceMatrix[source][3] += 1


def buildReviewHelper(articles, sources, claim, cred=None):
	if claim is None:
		return
	review = Review(claim, cred)
	posTK10 = topK(10)
	negTK10 = topK(10)
	articlesScore = []
	# book keeping to  
	relatedSnippets = []
	posStanceScores = []
	negStanceScores = []
	posTKrelatedSnippets = []
	negTKrelatedSnippets = []
	
	for article, source in zip(articles, sources):
		_, relatedSnippetsX_, relatedSnippets_, _, overlapScores_ = rsExtractor.extract(claim, article)
		# can be many other edge cases
		if relatedSnippets_ is not None:
			posTK3 = topK(3)
			negTK3 = topK(3)
			stanceProb_ = stanceClf.predict_porb(relatedSnippetsX_)
			del relatedSnippetsX_
			stanceScore_ = stanceProb_ * overlapScores_
			posTK10.add(stanceScore_[:,0])
			negTK10.add(stanceScore_[:,1])
			posTK3.add(stanceScore_[:,0])
			negTK3.add(stanceScore_[:,1])
			articlesScore.append((posTK3.avg(), negTK3.avg()))
			updateSource ((posTK10.avg(), negTK10.avg()), source, cred)
			relatedSnippets.extend(relatedSnippets_)
			del relatedSnippets_
			posStanceScores.extend(list(stanceScore_[:,0]))
			negStanceScores.extend(list(stanceScore_[:,1]))  

	review.addArticles(articlesScore, source)
	for score in posTK10.nums:
		posTKrelatedSnippets.append(relatedSnippets[posStanceScores.index(score)])
	for score in negTK10.nums:
		negTKrelatedSnippets.append(relatedSnippets[negStanceScores.index(score)])
	review.addSnippets(posTKrelatedSnippets, negTKrelatedSnippets)
	'''
	### CHEAT ###
	posScore = sum([score[0] for score in articlesScore])
	negScore = sum([score[1] for score in articlesScore])
	score = max(2.4*posScore, negScore) / len(articlesScore)
	# pos
	print (claim)
	if (2.4*posScore > negScore):
		print ('label: '+str(cred)+' predict: '+'0'+' confidence: '+str(score))
		print (posTKrelatedSnippets) 
	else:
		print ('label: '+str(cred)+' predict: '+'1'+' confidence: '+str(score))
		print (negTKrelatedSnippets) 
	'''
	reviews.append(review)


def buildReview(i):
	filePath = filePaths[i]
	if filePath == '.DS_Store':
		return
	print(filePath)
	articles, sources = reader.readGoogle(filePath)
	claim, cred = reader.readSnopes(filePath)
	buildReviewHelper(articles, sources, claim, cred)


def main():
	global sourceMatrix
	global reviews
	print ('start claim evaluation ...')
	if not os.path.isfile(reviewPath):
		for i in range(int(sys.argv[9]), int(sys.argv[10])):
			buildReview(i)
		
		'''
		pool = Pool(processes=8)
		pool.map(buildReview, [i for i in range(len(filePaths))])
		pool.join()
		'''
		#print list(reviews)
		print dict(sourceMatrix)
		pickle.dump(list(reviews), io.open(reviewPath, 'wb'))
		pickle.dump(dict(sourceMatrix), io.open(sourcePath, 'wb'))
	else:
		reviews = pickle.load(io.open(reviewPath, 'rb'))
		sourceMatrix = pickle.load(io.open(sourcePath, 'rb'))






if __name__ == '__main__':
	main()

