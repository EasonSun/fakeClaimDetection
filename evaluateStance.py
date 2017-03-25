import numpy as np
import os
import re
from extractLGfeatures import extractLGfeatures

# hyper parameter
overlapThreshold = .018
MIN_DF = .008

def extractSnippets(article):
    snippets = []
    snippetMarkNumbers = []	#list of list, inner list records number of ! ? ""
    NSS = 4 # number of a snippet in a sentence
    #articleSentences = re.split(r'[.|!|?]', article)
    articleSentences = []
    articleSentencesWithMarks = []
    sentence = ''
    for c in article:
    	if c not in ['?', '!', '.']:
    		sentence += c
    	else:
    		articleSentences.append(sentence)
    		if c in ['?', '!']:
    			sentence += c
    		else:
    			sentence += ' '
    		articleSentencesWithMarks.append(sentence)
    		sentence = ''
    # print(articleSentences)
    # no stripped kinda needed: like $100, "bill", these are critical
    # but vocab from the vectorizer is without these 
    for i in range(len(articleSentences)):
    	articleSentences[i] = " ".join(re.findall("[a-zA-Z0-9]+", articleSentences[i])) + ' '
    # print(articleSentences)
    while '' in articleSentences:
        articleSentences.remove('')
    while '' in articleSentencesWithMarks:
        articleSentencesWithMarks.remove('')
    for i in range(len(articleSentences) - NSS + 1):
        temp = articleSentences[i : i+NSS]
        snippet = ""
        for j in range(NSS):
            snippet = snippet + temp[j]
        snippets.append(snippet)

        snippetMarkNumber = [0,0,0]
        for j in range(NSS):
        	curSentence = articleSentencesWithMarks[i+j]
        	if '!' in curSentence:
        		snippetMarkNumber[0] = 1
        	if '?' in curSentence:
        		snippetMarkNumber[1] = 1
        	if '"' in curSentence:
        		snippetMarkNumber[2] = 1

        snippetMarkNumbers.append(snippetMarkNumber)

    return snippets, snippetMarkNumbers

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
	relatedSnippetMarkNumbers = []	# record number of ! ? "", list of list

	from sklearn.feature_extraction.text import CountVectorizer
	# empty string can be taken as all 0 vectors
	# using both uni- and bi-grams
	vectorizer = CountVectorizer(analyzer = "word", \
								preprocessor = None, \
								stop_words = 'english', \
								ngram_range=(1, 2))
	                             #max_features = 5000) 

	from sklearn.metrics.pairwise import cosine_similarity

	# claim: str, article: str, articleLabel: int
	for claim, article, articleLabel in zip(claims, articles, articleLabels):
		#print (article)
		# print (claim)
		snippets, snippetMarkNumbers = extractSnippets(article)
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
		relatedSnippetMarkNumbers.extend([snippetMarkNumbers[i] for i in overlapIdx])
		#print(relatedSnippet)
		#print(relatedSnippetLabels)
		'''
		# printStat
		print (claim)
		print(claimX.shape)
		print(snippetsX.shape)
		print (minSimilarityScore)
		'''
	return relatedSnippet, relatedSnippetLabels, relatedSnippetMarkNumbers

def evaluateHelper(X, y):
	print("X dim and y dim: ")
	print(X.shape, y.shape)	#3227, 2929
	ratioImbalance = np.sum(y) / (y.shape - np.sum(y))
	print ('ratio of imbalance, positive : negative is %4f' %ratioImbalance)
	print ("overlapThreshold = %f" %overlapThreshold)
	print("MIN_DF = %f" %MIN_DF)

	resultFile = open('stanceResult.txt', 'w')
	resultFile.write("X dim and y dim: \n")
	nunSample, numFeature = X.shape	
	resultFile.write('%i, %i \n' %(nunSample, numFeature))
	resultFile.write ('ratio of imbalance, positive : negative is %4f \n' %ratioImbalance)
	resultFile.write ("overlapThreshold = %f \n" %overlapThreshold)
	resultFile.write("MIN_DF = %f \n" %MIN_DF)

	
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt')

	from sklearn.model_selection import GridSearchCV
	# grid = dict(n_estimators=[500, 1000], max_depth=[2000, 2500])
	grid = dict(n_estimators=[500], max_depth=[2000])
	forestGS = GridSearchCV(estimator=forest, param_grid=grid, cv=5)
	forestGS.fit(X, y)

	means = forestGS.cv_results_['mean_test_score']
	stds = forestGS.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, forestGS.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		resultFile.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))


def evaluateStance(claims, articles, articleLabels, isFeatureGenerated=False):
	relatedSnippetX = np.array(0)
	relatedSnippet_y = np.array(0)
	relatedSnippetMarkNumberX = np.array(0)
	if not isFeatureGenerated:
		relatedSnippet, relatedSnippetLabels, relatedSnippetMarkNumbers = extractRelatedSnippet(claims, articles, articleLabels)
		from sklearn.feature_extraction.text import CountVectorizer
		# empty string can be taken as all 0 vectors
		# using both uni- and bi-grams
		vectorizer = CountVectorizer(analyzer = "word",   \
		                             tokenizer = None,    \
		                             preprocessor = None, \
		                             ngram_range=(1, 2), \
		                             min_df=MIN_DF)	
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

		relatedSnippetMarkNumberX = np.array(relatedSnippetMarkNumbers)
		np.save('relatedSnippetMarkNumberX', relatedSnippetMarkNumberX)

		# print("relatedSnippetX dim and relatedSnippet_y dim: ")
		# print(relatedSnippetX.shape, relatedSnippet_y.shape)

	else:
		relatedSnippetX = np.load('relatedSnippetX.npy')
		relatedSnippet_y = np.load('relatedSnippet_y.npy')
		relatedSnippetMarkNumberX = np.load('relatedSnippetMarkNumberX.npy')

	evaluateHelper(relatedSnippetX, relatedSnippet_y)
	'''
	RESULTS
	0.757 (+/-0.019) for {'max_depth': 2000, 'n_estimators': 500}
	0.753 (+/-0.020) for {'max_depth': 2000, 'n_estimators': 1000}
	0.755 (+/-0.018) for {'max_depth': 2500, 'n_estimators': 500}
	0.756 (+/-0.016) for {'max_depth': 2500, 'n_estimators': 1000}
	'''

	'''
	# concat ! ? " marks 
	relatedSnippetX = np.concatenate((relatedSnippetX, relatedSnippetMarkNumberX), axis=1)
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt')
	from sklearn.model_selection import GridSearchCV
	grid = dict(n_estimators=[500], max_depth=[2000])
	forestGS = GridSearchCV(estimator=forest, param_grid=grid, cv=5)
	forestGS.fit(relatedSnippetX, relatedSnippet_y)

	means = forestGS.cv_results_['mean_test_score']
	stds = forestGS.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, forestGS.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	'''

	'''
	RESULTS
	dons't work
	'''

def evaluateStanceLg(claims, articles, articleLabels, lgFeaturesPath, isFeatureGenerated=False):
	relatedSnippetLgX = np.array(0)
	relatedSnippetLg_y = np.array(0)
	relatedSnippetMarkNumberX = np.array(0)
	if not isFeatureGenerated:
		relatedSnippet, relatedSnippetLabels, relatedSnippetMarkNumbers = extractRelatedSnippet(claims, articles, articleLabels)
		relatedSnippetLg = extractLGfeatures(relatedSnippet, lgFeaturesPath)
		# too many features!
		from sklearn.feature_extraction.text import CountVectorizer
		vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             #keep !, ?, ', and " as features
                             token_pattern = '(?u)\\b\\w\\w+\\b|!|\\?|\\"|\\\'', \
                             stop_words = None) # no need to strip off stop_words because all linguistic features
		 
		relatedSnippetLgX = vectorizer.fit_transform(relatedSnippet)
		print (vectorizer.vocabulary_)
		return
		relatedSnippetLgX = relatedSnippetLgX.toarray()
		np.save('relatedSnippetLgX', relatedSnippetLgX)

		relatedSnippetLg_y = np.array(relatedSnippetLabels)
		np.save('relatedSnippetLg_y', relatedSnippetLg_y)

	else:
		relatedSnippetLgX = np.load('relatedSnippetLgX.npy')
		relatedSnippetLg_y = np.load('relatedSnippetLg_y.npy')
	
	evaluateHelper(relatedSnippetLgX, relatedSnippetLg_y)
