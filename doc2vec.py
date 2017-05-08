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

def selfSimilarityAccuracy(model, ctr):
	ranks = []
	for doc_id in range(ctr):
	    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
	    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
	    rank = [docid for docid, sim in sims].index(doc_id)
	    ranks.append(rank)
	rankSummary = collections.Counter(ranks)
	return (rankSummary[0]+rankSummary[1]+rankSummary[2])/ctr

ctr = 0

for filePath in os.listdir(googleDataPath):
	articles_, _ = readGoogle(filePath)
	if articles_ == [""]:
		continue
	train_corpus.extend(list(formatArticle(articles_)))
	if ctr%5000 == 0:
		pickle.dump(train_corpus, open(outDataPath+ctr, 'wb'))
		del train_corpus

print (str(ctr)+' articles loaded')
#print (train_corpus[1])

model = gensim.models.doc2vec.Doc2Vec(size=300, min_count=3, window=3)
model.build_vocab(train_corpus)
print (len(model.wv.vocab))
#training can vary a little becasue random seeds.
prevRankAccu = 0
iter = 5
while(True):
	model.train(train_corpus, total_examples=model.corpus_count, epochs=iter)
	rankAccu = selfSimilarityAccuracy(model, ctr)
	if (rankAccu < prevRankAccu + .01):
		model.train(train_corpus, total_examples=model.corpus_count, epochs=iter-2)
		break
	prevRankAccu = rankAccu
	iter += 2

#model.iter = iter
print (prevRankAccu)
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model.save(doc2vecModelPath)
'''
r1 = print (model.n_similarity("i am a boy".split(), "she is a girl".split()))


model1 = gensim.models.doc2vec.Doc2Vec.load(doc2vecModelPath)
# use this model1.wv.vocab to retrive log 

r2 = print (model.n_similarity("i am a boy".split(), "she is a girl".split()))
assert(r1 == r2)

'''


