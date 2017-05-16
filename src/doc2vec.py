from gensim import models
import re
import os
import time
import json
import io
import numpy

#googleDataPath="../data/Google_test"
#doc2vecModelPath="../data/doc2vec_apnews.bin"
googleDataPath="../data/Google_test"
doc2vecModelPath = '../data/doc2vec_apnews.bin'

contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

alteos = re.compile(r'([!\?])')

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
	print (filePath)
	data = json.load(io.open(filePath, 'r', encoding='utf-8', errors='ignore'))
	snippets = []
	for article in data['article']:
		# article = cleanText(article)
		articleSentences = alteos.sub(r' \1 .', article).rstrip("(\.)*\n").split('.')
		ctr1 = 0

		snippet = ''
		for sen in articleSentences:
			if (len(sen.split())) > 3:
				sen = cleanText(sen)
				if (len(sen.split())) > 3:
					if ctr1 < 3:
						snippet += (' ' + sen)
						ctr1 += 1

			if ctr1 == 3:
				snippets.append(snippet.split())
				ctr1 = 0
				snippet = ''
	return snippets
#inference hyper-parameters
start_alpha=0.01
infer_epoch=1000

#load model
#m = models.KeyedVectors.load_word2vec_format(doc2vecModelPath, binary=True)
m = models.Doc2Vec.load(doc2vecPath)
for filePath in os.listdir(googleDataPath):
	print (filePath)
	if not filePath.endswith('.json'):
		continue
	t1 = time.time()
	snippets = readGoogle(filePath)
	t2 = time.time()
	print (t2 - t1)
	print (len(snippets))
	for snippet in snippets:
		t1 = time.time()
		vec = m.infer_vector(snippet, alpha=start_alpha, steps=infer_epoch)
		t2 = time.time()
		print (vec.shape)
		print (t2 - t1)
		break
