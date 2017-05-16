import os
import json
import matplotlib.pyplot as plt
import pickle
import re
from multiprocessing import Pool, Manager, Process
import time


googleDataPath="data/Google_processed"
contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

alteos = re.compile(r'([!\?])')

manager = Manager()
articleNumSenDist = {}
snippetLenDist3 = manager.dict()
snippetLenDist4 = manager.dict()


def cleanText(text):        
    # cleaner (order matters)
    text = text.lower()
    text = contractions.sub('', text)
    text = symbols.sub(r' \1 ', text)
    text = singles.sub(' ', text)
    text = seps.sub(' ', text)
    return text

def countNumSen(filePath):
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


def countNumWord(filePath):
	filePath = os.path.join(googleDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	for article in data['article']:
		# article = cleanText(article)
		articleSentences = alteos.sub(r' \1 .', article).rstrip("(\.)*\n").split('.')
		ctr1 = 0
		ctr2 = 0
		snippet = ''
		for sen in articleSentences:
			if (len(sen.split())) > 3:
				sen = cleanText(sen)
				if (len(sen.split())) > 3:
					if ctr1 < 3:
						snippet += (' ' + sen)
						ctr1 += 1
					if ctr2 < 4:
						snippet += (' ' + sen)
						ctr2 += 1

			if ctr1 == 3:
				snippetLen = len(snippet.split())
				if snippetLen in snippetLenDist3:
					snippetLenDist3[snippetLen] += 1
				else:
					snippetLenDist3[snippetLen] = 1
				ctr1 = 0
				snippet = ''
			elif ctr2 == 4:
				snippetLen = len(snippet.split())
				if snippetLen in snippetLenDist4:
					snippetLenDist4[snippetLen] += 1
				else:
					snippetLenDist4[snippetLen] = 1
				ctr2 = 0
				snippet = ''


filePaths = os.listdir(googleDataPath)
def count(i):
	print ('here')
	filePath = filePaths[i]
	if filePath == '.DS_Store':
		return
	#countNumSen(filePath)
	countNumWord(filePath)

'''
def count(snippetLenDist3, snippetLenDist4):
	for filePath in os.listdir(googleDataPath):
		if filePath == '.DS_Store':
			return
		#countNumSen(filePath)
		countNumWord(filePath, snippetLenDist3, snippetLenDist4)
'''

# has to use a iterative scenario, otherwise all the workers do a copy of everything

t1 = time.time()
pool = Pool(processes=4)
pool.starmap(count, [(i,) for i in range(len(filePaths))])
t2 = time.time()
print (t2-t1)


# garanteed to be on different process
'''
for i in range(4):
    p = Process(target=count, args=(snippetLenDist3, snippetLenDist4,))
    p.start()
    p.join()
'''

def save(dist):
	pickle.dump(dict(eval(dist)), open(dist, 'wb'))	#converted to pure dict
'''
t1 = time.time()
save('snippetLenDist3')
save('snippetLenDist4')
t2 = time.time()
print (t2-t1)
'''

def load(dist):
	return pickle.load(open(dist, 'rb'))


t1 = time.time()
snippetLenDist3 = load('snippetLenDist3')
#snippetLenDist4 = load('snippetLenDist4')
t2 = time.time()
print (t2-t1)


def plot(dist):
	plt.bar(range(len(eval(dist))), eval(dist).values(), color='g')
	plt.title(dist)
	plt.xlabel('Number of Words in a Snippet')
	plt.ylabel('Number')
	plt.xlim([0,400])
	plt.ylim([0,450000])
	plt.savefig(dist + '.png')

'''
t1 = time.time()
#plot('snippetLenDist3')
plot('snippetLenDist4')
t2 = time.time()
print (t2-t1)
'''

def stat(dist):
	dist = eval(dist)
	summ = 0
	count = 0
	maxArticleCount = 0
	for key in dist.keys():
		if dist[key] > maxArticleCount:
			maxArticleCount = dist[key]
		summ += dist[key]*key
		count += dist[key]
	print (summ)
	print ('average')
	print (summ/count)
	print (maxArticleCount)

stat('snippetLenDist3')