import os
import sys 

def extractLGfeatures(claims, lgFeaturesPath):
	allFeaturePath = os.path.join(lgFeaturesPath, 'allFeatures.txt')
	if not (os.path.isfile(allFeaturePath)):
		print ("Creating file allFeatures.txt to group all linguistic features")
		afFile = open(allFeaturePath, 'w')
		for filePath in os.listdir(lgFeaturesPath):
			#print filePath
			if filePath != 'allFeatures.txt' and filePath.endswith('.txt'):
				print (filePath)
				curFile = open(os.path.join(lgFeaturesPath, filePath), 'r', encoding='utf-8', errors='ignore')
				writeAFFile(afFile, curFile)
				curFile.close()
		afFile.close()
	

	afFile = open(allFeaturePath, 'r')
	allFeatures = set(afFile.read().split())
	afFile.close()
	return helper(claims, allFeatures)

def writeAFFile(afFile, curFile):
	# parse possible "#... ...#" comment
	line = curFile.readline()
	if (line[:3] == '###'):
		line = curFile.readline()
		while(line[:3] != '###'):
			line = curFile.readline()
		line = curFile.readline()
	else:
		afFile.write(line)
	for line in curFile:
		afFile.write(line)
	afFile.write('\n')

# Many synonms (noun/verb) get messed up
def helper(claims, allFeatures):
	import re
	lgClaims = []
	for claim in claims:
		lgClaim = []
		words = re.findall(r"[\w']+|[!?\"]", claim)	# keep ! ? "
		# words = re.findall("[a-zA-Z]+", claim)
		for word in words:
			if word in allFeatures:
				lgClaim.append(word)

		lgClaims.append(' '.join(lgClaim))
	return lgClaims