import gensim
import os
import collections
import random
import json
import pickle


# on server
googleDataPath="../data/Google_processed"
doc2vecModelPath="../data/doc2vecModel"
outDataPath="../data/trainedCorpus"

'''
googleDataPath="data/Google_test"
doc2vecModelPath="data/doc2vecModel"
outDataPath="data/trainedCorpus"
'''

train_corpus = []

def readGoogle(filePath):
	filePath = os.path.join(googleDataPath, filePath)
	data = json.load(open(filePath, 'r', encoding='utf-8', errors='ignore'))
	return data['article'], data['source']

def senList(article):
	import re
	contractions = re.compile(r"'|-|\"")
	# all non alphanumeric
	symbols = re.compile(r'(\W+)', re.U)
	# single character removal
	singles = re.compile(r'(\s\S\s)', re.I|re.U)
	# separators (any whitespace)
	seps = re.compile(r'\s+')

	# cleaner (order matters)
	def clean(text): 
	    text = text.lower()
	    text = contractions.sub('', text)
	    text = symbols.sub(r' \1 ', text)
	    text = singles.sub(' ', text)
	    text = seps.sub(' ', text)
	    return text

	# sentence splitter
	alteos = re.compile(r'([!\?])')
	def sentences(l):
	    l = alteos.sub(r' \1 .', l).rstrip("(\.)*\n")
	    return l.split('.')

	article_ = [clean(s) for s in sentences(article)]
	article_ = [x.split() for x in article_ if len(x.split())>1]
	return [item for sublist in article_ for item in sublist]

# articles: a list of str, each str is an article
def formatArticle(articles):
	global ctr
	for article in articles:
		articleList = senList(article)
		yield gensim.models.doc2vec.TaggedDocument(articleList, [ctr])
		ctr += 1

def selfSimilarityAccuracy(model):
	ranks = []
	numArticle = len(train_corpus)
	#for doc_id in range(numArticle):
	numSame = 0
	for article in train_corpus:
	    inferred_vector = model.infer_vector(article.words)
	    sims = model.docvecs.most_similar([inferred_vector], topn=1)
	    #print (article.tags[0], sims[0][0])
	    if article.tags[0] == sims[0][0]:
	    	numSame += 1
	    #rank = [docid for docid, sim in sims].index(doc_id)
	    #ranks.append(rank)
	#rankSummary = collections.Counter(ranks)
	#print (rankSummary)
	#return (rankSummary[0]+rankSummary[1]+rankSummary[2])/numArticle
	return numSame / numArticle

ctr = 0
chunk = 1000
numChunk = 1
'''
for filePath in os.listdir(googleDataPath):
	if filePath == '.DS_Store':
		continue
	articles_, _ = readGoogle(filePath)
	if articles_ == [""]:
		continue
	train_corpus.extend(list(formatArticle(articles_)))
	if ctr >= chunk*numChunk and ctr != 0:
		print (numChunk)
		pickle.dump(train_corpus, open(outDataPath+str(numChunk), 'wb'))
		del train_corpus
		train_corpus = []
		numChunk += 1

if (train_corpus != []):
	pickle.dump(train_corpus, open(outDataPath+str(numChunk), 'wb'))
	del train_corpus
	train_corpus = []
'''
for filePath in os.listdir(googleDataPath):
	if filePath == '.DS_Store':
		continue
	articles_, _ = readGoogle(filePath)
	if articles_ == [""]:
		continue
	if ctr >= chunk*numChunk and ctr != 0:
		print (numChunk)
		numChunk += 1
'''
# cannot do this batch learning

'''

print (str(ctr)+' articles loaded')
#print (train_corpus[1])

model = gensim.models.doc2vec.Doc2Vec(size=300, min_count=3, window=3, workers=4)
for j in range(1, numChunk+1):
	with open(outDataPath+str(j), 'rb') as f:
		train_corpus.extend(pickle.load(f))
model.build_vocab(train_corpus)
del train_corpus
train_corpus = []
print (len(model.wv.vocab))
model.save(doc2vecModelPath)

#training can vary a little becasue random seeds.
prevRankAccu = 0
iter = 20
for ijk in range(15):
	for j in range(1, numChunk+1):
		train_corpus_ = pickle.load(open(outDataPath+str(j), 'rb'))
		model.train(train_corpus_, total_examples=model.corpus_count, epochs=iter)
	model.save(doc2vecModelPath)
	j = random.randint(1, numChunk)
	train_corpus = pickle.load(open(outDataPath+str(j), 'rb'))
	rankAccu = selfSimilarityAccuracy(model)
	#print (rankAccu)
	if (rankAccu != 0 and rankAccu < prevRankAccu + .01):
		model.train(train_corpus, total_examples=model.corpus_count, epochs=iter-2)
		break
	prevRankAccu = rankAccu
	iter += 2

#model.iter = iter
print (prevRankAccu)
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model.save(doc2vecModelPath)
print (model.n_similarity("i am a boy".split(), "she is a girl".split()))
'''
r1 = print (model.n_similarity("i am a boy".split(), "she is a girl".split()))


model1 = gensim.models.doc2vec.Doc2Vec.load(doc2vecModelPath)
# use this model1.wv.vocab to retrive log 

r2 = print (model.n_similarity("i am a boy".split(), "she is a girl".split()))
assert(r1 == r2)

'''


