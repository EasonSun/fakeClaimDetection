import os
import json

googleDataPath="data/Google_processed"

def readGoogle(filePath):
	filePath = os.path.join(googleDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	return len(data['article'])

numArticle = 0
for filePath in os.listdir(googleDataPath):
	if filePath == '.DS_Store':
		continue
	numArticle += readGoogle(filePath)

print (numArticle)