import os
import json
import matplotlib.pyplot as plt
import pickle
import re


googleDataPath="data/Google_processed"
contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

alteos = re.compile(r'([!\?])')

articleLenDist = {}

def cleanText(text):        
    # cleaner (order matters)
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

def readGoogle(filePath):
	filePath = os.path.join(googleDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	for article in data['article']:
		# article = cleanText(article)
		articleSentences = alteos.sub(r' \1 .', article).rstrip("(\.)*\n").split('.')
		articleSentences = [sen for sen in articleSentences if len(sen.split())>3]
		articleLen = len(articleSentences)
		if articleLen in articleLenDist:
			articleLenDist[articleLen] += 1
		else:
			articleLenDist[articleLen] = 1

'''
numArticle = 0
for filePath in os.listdir(googleDataPath):
	if filePath == '.DS_Store':
		continue
	readGoogle(filePath)

pickle.dump(articleLenDist, open('articleLenDist', 'wb'))
'''
articleLenDist = pickle.load(open('articleLenDist', 'rb'))
#plt.bar(list(articleLenDist.keys()), articleLenDist.values(), color='g')
plt.bar(range(len(articleLenDist)), articleLenDist.values(), color='g')
plt.title('Article Sentence Length Distribution')
plt.xlabel('Number of sentences')
plt.ylabel('Number of articles')
plt.ylim([0,500])
plt.savefig('articleLenDist.png')