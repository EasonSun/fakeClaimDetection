import sys
import os
import json
import numpy
import numpy as np


def readSnopes(dataPath, experimentPath):
	claimPath = experimentPath + 'claims.txt'
	articlePath = experimentPath + 'articles.txt'
	stancePath = experimentPath + 'stance'
	credPath = experimentPath + 'cred'
	if os.path.isfile(claimPath):
		return

	data = []
	for filePath in os.listdir(dataPath):
		if filePath.endswith('.json'):
			file = open(os.path.join(dataPath, filePath), 'r', encoding="utf-8")
			data.append(json.load(file))
			file.close()

	claimFile = open(claimPath, 'w')
	articleFile = open(articlePath, 'w')
	stanceList = []
	credList = []


	for item in data:	# each item is a dict
		if item['Description'] != '':
			claimFile.write("%s\n" % item['Claim'])
			articleFile.write('%s\n' % item['Description'])
			if item['Credibility'] in ['true', 'mostly true']: 
				stanceList.append(0)	# for
			elif item['Credibility'] in ['false', 'mostly false']: 
				stanceList.append(1) # against

		if item['Credibility'] in ['true', 'mostly true']: 
			credList.append(0)	# for
		elif item['Credibility'] in ['false', 'mostly false']: 
			credList.append(1) # against

	claimFile.close()
	articleFile.close()
	np.save(stancePath, np.array(stanceList))
	np.save(credPath, np.array(credList))

def readGoogle(dataPath, experimentPath):
	pass

def main():
	snopesPath = sys.argv[1]
	experimentPath = sys.argv[2]
	
	readSnopes(snopesPath, experimentPath)
	readGoogle(googlePath, experimentPath)

if __name__ == '__main__':
    main()