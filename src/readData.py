import sys
import os
import json
import numpy as np
import pickle


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

def readGlove(gloveFile, experimentPath):
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    pickle.dump(model, open(experimentPath+'glove', 'wb'))

def main():
	snopesPath = sys.argv[1]
	experimentPath = sys.argv[2]
	glovePath = sys.argv[3]

	readSnopes(snopesPath, experimentPath)
	#readGoogle(googlePath, experimentPath)
	readGlove(glovePath, experimentPath)

if __name__ == '__main__':
    main()