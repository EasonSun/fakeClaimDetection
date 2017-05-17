import numpy as np
import os
import sys
import json
import io
import pickle
import heapq
from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor

sourcePath = 1
doc2vecPath = 2
logPath = 3
experimentPath = 4
#rsExtractor = relatedSnippetsExtractor(overlapThreshold, doc2vecPath=doc2vecPath)
#stanceClf = Classifier('stance', logPath, experimentPath)

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
		return sum(self.nums)


class Review(object):
	"""docstring for ClassName"""
	def __init__(self, claim, label=None):
		self.claim = claim
		self.label = label
		#self.sourceMatrix = pickle.load(io.open(sourcePath, 'rb'))
		print sourcePath

	def addSnippets(self,  posTKrelatedSnippets, negTKrelatedSnippets):
		# top 10
		self.posTKrelatedSnippets = posTKrelatedSnippets
		self.negTKrelatedSnippets = negTKrelatedSnippets
		#self.snippetsScore = snippetsScore

	def addArticles(self, articlesScore, source):
		# number of articles
		self.articlesScore = articlesScore
		self.source = source
	
	def review(self, articles, sources):
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
				stanceProb_ = stanceClf.predict_porb(relatedSnippetsX_)
				del relatedSnippetsX_
				stanceScore_ = stanceProb_ * overlapScores_
				posTK10.add(stanceScore_[:,0])
				negTK10.add(stanceScore_[:,1])
				articlesScore.append((posTK10.avg(), negTK10.avg()))
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
		# print (posTKrelatedSnippets)  
		return review