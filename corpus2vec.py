import os 
import json

googleDataPath="data/Google_processed"
gloveDataPath="data/glove/glove.6B.200d.txt"

def readGoogle(filePath):
	filePath = os.path.join(googleDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	return data['article'], data['source']

def loadGloveModel(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    vocab = set()
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
        vocab.add(word)
    print "Done.",len(model)," words loaded!"
    return model, vocab

# for a list
def toVec(articles):
	vecs = []
	for article in articles:
		article_ = article.lower().split()
		for word in vocab:
			

model, vocab = loadGloveModel(gloveDataPath)
def main():
	for filePath in os.listdir(googleDataPath):
		articles_, _ = readGoogle(filePath)
		vecs = toVec(articles_)