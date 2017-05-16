import sys
import os
import json
import numpy as np
import pickle
import io


class ClaimReader(object):
	"""docstring for reader"""
	def __init__(self, snopeDataPath, googleDataPath):
		self.snopeDataPath = snopeDataPath
		self.googleDataPath = googleDataPath

	def readSnopes(self, filePath):
		filePath = os.path.join(self.snopeDataPath, filePath)
		data = json.load(io.open(filePath, 'r', encoding='utf-8', errors='ignore'))
		if data['Credibility'] in ['true', 'mostly true']:
			return data['Claim'], 0# for
		elif data['Credibility'] in ['false', 'mostly false']: 
			return data['Claim'], 1


	def readGoogle(self, filePath):
		filePath = os.path.join(self.googleDataPath, filePath)
		data = json.load(io.open(filePath, 'r', encoding='utf-8', errors='ignore'))
		return data['article'], data['source']

	def readGlove(gloveFile, experimentPath):
	    f = io.open(gloveFile,'r')
	    model = {}
	    for line in f:
	        splitLine = line.split()
	        word = splitLine[0]
	        embedding = np.array([float(val) for val in splitLine[1:]])
	        model[word] = embedding
	    print ("Done.",len(model)," words loaded!")
	    pickle.dump(model, io.open(experimentPath+'glove', 'wb'))