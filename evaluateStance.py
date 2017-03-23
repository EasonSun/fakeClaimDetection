import numpy as np
import os
import re


def extractSnippets(article, articleLabel):
    snippets = []
    snippetLabels = []
    NSS = 4 # number of a snippet in a sentence
    articleSentences = re.split(r'[.|!|?]', article)
    
    # no stripped kinda needed: like $100, "bill", these are critical
    # but vocab from the vectorizer is without these 
    for i in range(len(articleSentences)):
    	articleSentences[i] = " ".join(re.findall("[a-zA-Z0-9]+", articleSentences[i]))
    while '' in articleSentences:
        articleSentences.remove('')
    for i in range(len(articleSentences) - NSS + 1):
        temp = articleSentences[i : i+NSS]
        snippet = ""
        for j in range(NSS):
            snippet = snippet + temp[j]
        snippets.append(snippet)
        snippetLabels.append(articleLabel)
    return snippets, snippetLabels

def extractVocab(claims, snippets, vectorizer):
	result = {}
	try:
		raw1 = vectorizer.fit(claims).vocabulary_
		raw2 = vectorizer.fit(snippets).vocabulary_
	except ValueError:
		return {}
	# print (raw2)
	nextValue = 0
	for key, value in raw1.items():
		if key not in result.keys():
			result[key] = nextValue
			nextValue += 1
			# print (key)
	for key, value in raw2.items():
		if key not in result.keys():
			result[key] = nextValue
			nextValue += 1
			# print (key)
	return result

def extractRelatedSnippet(claims, articles, articleLabels):
	relatedSnippet = []
	relatedSnippetLabels = []
	overlapThreshold = .02

	from sklearn.feature_extraction.text import CountVectorizer
	# empty string can be taken as all 0 vectors
	# using both uni- and bi-grams
	vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = 'english', \
	                             ngram_range=(1, 2))
	                             #max_features = 5000) 

	from sklearn.metrics.pairwise import cosine_similarity

	# claim: str, article: str, articleLabel: int
	for claim, article, articleLabel in zip(claims, articles, articleLabels):
		#print (article)
		# print (claim)
		snippets, snippetLabels = extractSnippets(article, articleLabels)
		# find vocab for this pair so as to do vector similarity
		vectorizer.vocabulary = None
		vocab = extractVocab([claim], snippets, vectorizer)
		if len(vocab.keys()) == 0:
			# bad thing can happen
			continue 
		# print (vocab)
		vectorizer.vocabulary = vocab
		# assert(vectorizer.vocabulary == vocab)
		claimX = vectorizer.fit_transform([claim])
		claimX = claimX.toarray()
		# print(claimX.shape)
		# print(claimX[0][210])
		snippetsX = vectorizer.fit_transform(snippets)
		snippetsX = snippetsX.toarray()
		# print(snippetsX.shape)
		# print (snippetsX[-1][209])

		similarityScore = cosine_similarity(claimX, snippetsX)
		if (np.count_nonzero(similarityScore) == 0):
			# bad and weird thing happens 
			continue
		minSimilarityScore = np.min(similarityScore[np.nonzero(similarityScore)])
		if (minSimilarityScore < overlapThreshold):
			continue
		# print (minSimilarityScore)
		overlapIdx = np.where(cosine_similarity(snippetsX, claimX) > overlapThreshold)[0]
		#print (overlapIdx)
		relatedSnippetLabels.extend([articleLabel for i in range(len(overlapIdx))])
		snippets = np.array([[snippet] for snippet in snippets])
		#print (snippets.shape)
		# from vector back to sentence to later use them in the same feature space
		relatedSnippet.extend([''.join(snippet) for snippet in snippets[overlapIdx].tolist()])
		#print(relatedSnippet)
		#print(relatedSnippetLabels)
		'''
		# printStat
		print (claim)
		print(claimX.shape)
		print(snippetsX.shape)
		print (minSimilarityScore)
		'''
	return relatedSnippet, relatedSnippetLabels

def evaluateStance(claims, articles, articleLabels, isFeatureGenerated=False):
	relatedSnippetX = np.array(0)
	relatedSnippet_y = np.array(0)
	if not isFeatureGenerated:
		relatedSnippet, relatedSnippetLabels = extractRelatedSnippet(claims, articles, articleLabels)
		from sklearn.feature_extraction.text import CountVectorizer
		# empty string can be taken as all 0 vectors
		# using both uni- and bi-grams
		vectorizer = CountVectorizer(analyzer = "word",   \
		                             tokenizer = None,    \
		                             preprocessor = None, \
		                             ngram_range=(1, 2), \
		                             min_df=.005)	
		'''
		the min df above is really important as the first step for fieature engineering
		.005 means only keep features apper more than .005 portion of docs
		that is roughly 486 docs
		'''
		relatedSnippetX = vectorizer.fit_transform(relatedSnippet)
		relatedSnippetX = relatedSnippetX.toarray()
		np.save('relatedSnippetX', relatedSnippetX)

		relatedSnippet_y = np.array(relatedSnippetLabels)
		np.save('relatedSnippet_y', relatedSnippet_y)

		# print("relatedSnippetX dim and relatedSnippet_y dim: ")
		# print(relatedSnippetX.shape, relatedSnippet_y.shape)

	else:
		relatedSnippetX = np.load('relatedSnippetX.npy')
		relatedSnippet_y = np.load('relatedSnippet_y.npy')

	print("relatedSnippetX dim and relatedSnippet_y dim: ")
	print(relatedSnippetX.shape, relatedSnippet_y.shape)	#3227, 2929
	ratioImbalance = np.sum(relatedSnippet_y) / (relatedSnippet_y.shape - np.sum(relatedSnippet_y))
	print ('ratio of imbalance, positive : negative is %4f' %ratioImbalance)
	# print (np.average(relatedSnippetX))	#0.00171037311133
	# print (np.percentile(relatedSnippetX, 3))


	'''
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt')

	from sklearn.model_selection import GridSearchCV
	grid = dict(n_estimators=[500, 1000], max_depth=[2000, 2500])
	forestGS = GridSearchCV(estimator=forest, param_grid=grid, cv=5)
	forestGS.fit(relatedSnippetX, relatedSnippet_y)

	means = forestGS.cv_results_['mean_test_score']
	stds = forestGS.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, forestGS.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	'''


	'''
	RESULTS
	only BoW: 
	0.757 (+/-0.019) for {'max_depth': 2000, 'n_estimators': 500}
	0.753 (+/-0.020) for {'max_depth': 2000, 'n_estimators': 1000}
	0.755 (+/-0.018) for {'max_depth': 2500, 'n_estimators': 500}
	0.756 (+/-0.016) for {'max_depth': 2500, 'n_estimators': 1000}
	'''
	