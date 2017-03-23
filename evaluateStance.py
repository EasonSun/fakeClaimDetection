import numpy as np
import os
import re


def extractSnippets(article, articleLabel):
    snippets = []
    snippetLabels = []
    #splittedArticle = re.split(r'[^a-zA-Z|^ ]', article)
    splittedArticle = re.split(r'[.|!|?]', article)
    for i in range(len(splittedArticle)):
        splittedArticle[i] = re.sub("[^a-zA-Z|^ ]+","",splittedArticle[i])
    while '' in splittedArticle:
        splittedArticle.remove('')
    q = len(splittedArticle) - 3
    for i in range(q):
        temp = splittedArticle[i:i+3]
        snippet = "";
        for j in range(3):
            snippet = snippet + temp[j]
        snippets.append(snippet)
        snippetLabels.append(articleLabel)
    return snippets, snippetLabels
    

def extractVocab(claims, snippets, vectorizer):
	result = {}
	raw1 = vectorizer.fit(claims).vocabulary_
	raw2 = vectorizer.fit(snippets).vocabulary_
	nextValue = 0
	for key, value in raw1.items():
		if key not in result.keys():
			result[key] = nextValue
			nextValue += 1
	for key, value in raw2.items():
		if key not in result.keys():
			result[key] = nextValue
			nextValue += 1
	return result

def extractRelatedSnippet(claims, articles, articleLabels):
	relatedSnippet = []
	relatedSnippetLabels = []
	overlapThreshold = .15

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

	for claim, article, articleLabel in zip(claims, articles, articleLabels):
		snippets, snippetLabels = extractSnippets(article, articleLabels)
		# find vocab for this pair so as to do vector similarity
		vocab = extractVocab([claim], snippets, vectorizer)
		# print (vocab)
		vectorizer.vocabulary = vocab
		claimX = vectorizer.fit_transform([claim])
		claimX = claimX.toarray()
		# print(claimX.shape)
		print(snippets)
		snippetsX = vectorizer.fit_transform(snippets)
		snippetsX = snippetsX.toarray()
		# print(snippetsX.shape)
		# print (cosine_similarity(claimX, snippetsX))
		overlapIdx = np.where(cosine_similarity(snippetsX, claimX) > overlapThreshold)[0]
		print (overlapIdx)
		relatedSnippetLabels.extend([articleLabel for i in range(len(overlapIdx))])
		snippets = np.array([[snippet] for snippet in snippets])
		#print (snippets.shape)
		# from vector back to sentence to later use them in the same feature space
		relatedSnippet.extend([''.join(snippet) for snippet in snippets[overlapIdx].tolist()])
		print(relatedSnippet)
		print(relatedSnippetLabels)
		break
	return relatedSnippet, relatedSnippetLabels 
 
def extractOverlapSnippet(claims, articles, articleLabels):
	relatedSnippetX = []
	relatedSnippet_y = []

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

	for claim, article, articleLabel in zip(claims, articles, articleLabels):
		snippets, snippetLabels = extractSnippets(article, articleLabels)
		print (len(snippets))
		# find vocab for this pair so as to do vector similarity
		vocab = extractVocab([claim], snippets, vectorizer)
		# print (vocab)
		vectorizer.vocabulary = vocab
		claimX = vectorizer.fit_transform([claim])
		claimX = claimX.toarray()
		# print(claimX.shape)
		snippetsX = vectorizer.fit_transform(snippets)
		snippetsX = snippetsX.toarray()
		# print(snippetsX.shape)
		# print (cosine_similarity(claimX, snippetsX))
		overlapIdx = np.where(cosine_similarity(snippetsX, claimX) > .15)[0]
		print (overlapIdx)
		relatedSnippet_y.extend([articleLabel for i in range(len(overlapIdx))])
		snippets = np.array([[snippet] for snippet in snippets])
		#print (snippets.shape)
		print (snippets[overlapIdx])
		relatedSnippetX.extend(snippets[overlapIdx].tolist())
		break
	return relatedSnippetX, relatedSnippet_y

def evaluateStance(claims, articles, articleLabels, isFeatureGenerated = False):
	relatedSnippetX = np.array(0)
	relatedSnippet_y = np.array(0)
	claims =re.sub("[^a-zA-Z|^ ]+","",claims)
	if not isFeatureGenerated:
		relatedSnippet, relatedSnippetLabels = extractRelatedSnippet(claims, articles, articleLabels)
		from sklearn.feature_extraction.text import CountVectorizer
		# empty string can be taken as all 0 vectors
		# using both uni- and bi-grams
		vectorizer = CountVectorizer(analyzer = "word",   \
		                             tokenizer = None,    \
		                             preprocessor = None, \
		                             ngram_range=(1, 2))
		relatedSnippetX = vectorizer.fit_transform([relatedSnippet])
		relatedSnippetX = relatedSnippetX.toarray()
		relatedSnippet_y = np.array(relatedSnippetLabels)
		print("relatedSnippetX dim and relatedSnippet_y dim: ")
		print(relatedSnippetX.shape, relatedSnippet_y.shape)
		#np.save('relatedSnippet', relatedSnippet)
		#np.save('relatedSnippetLabels', relatedSnippetLabels)

	else:
		relatedSnippetX = np.load('relatedSnippetX.npy')
		relatedSnippet_y = np.load('relatedSnippet_y.npy')

