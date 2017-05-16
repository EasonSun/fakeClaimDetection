import sys
import os
import json
import numpy as np
import pickle
import io

class StanceReader(object):
	"""docstring for StanceReader"""
	def __init__(self, snopeDataPath):
		self.snopeDataPath = snopeDataPath

	def readSnopes(self, filePath):
		file = io.open(os.path.join(self.snopeDataPath, filePath), 'r', encoding="utf-8")
		data = json.load(file)
		file.close()
		if data['Description'] != '':
			if data['Credibility'] in ['true', 'mostly true']:
				return data['Claim'], data['Description'], 0
			elif data['Credibility'] in ['false', 'mostly false']: 
				return data['Claim'], data['Description'], 1
		return None, None, None