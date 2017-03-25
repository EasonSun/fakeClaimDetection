'''
read in claims
write out linguistic feature for each claim
Qi will handle this 
-----------------
read in linguistic feature
do 10-fold cross validation
-----------------
read in articles related to claims
write out average stance scores for each claim
-----------------
read in stance scores 
do 10-fold cross validation
-----------------
combine linguistic and stance features - how?
do 10-fold cross validation 
-----------------

source credibility: does not understand 
-----------------

'''
import sys
import os
import json
import numpy
from extractLGfeatures import extractLGfeatures
from evaluateLG import evaluateLG
from evaluateStance import evaluateStance, evaluateStanceLg

dataPath = sys.argv[1]
lgFeaturesPath = sys.argv[2]

data = []
claims = []
labels = []
articles = []
articleLabels = []

'''
with open(dataPath, "r") as inFile:
	data = json.load(inFile)

# for our data
for item in data:
	claims.append(item['claims'])
	if item['rating'] == 'mostly true' | item['rating'] == 'true': 
		labels.append(0)
	elif item['rating'] == 'mostly false' | item['rating'] == 'false': 
		labels.append(1)
	else:
		labels.append(2)
'''
for filePath in os.listdir(dataPath):
	if filePath.endswith('.json'):
		file = open(os.path.join(dataPath, filePath), 'r')
		data.append(json.load(file))

for item in data:	# each item is a dict
	# claim
	claims.append(item['Claim'])
	# label
	if item['Credibility'] in ['true', 'mostly true']: 
		labels.append(0)
	elif item['Credibility'] in ['false', 'mostly false']: 
		labels.append(1)
	else:
		print (item['Claim_ID'], item['Credibility'])
	# descriptions
	if item['Description'] != '':
		articles.append(item['Description']) 
		if item['Credibility'] in ['true', 'mostly true']: 
			articleLabels.append(0)	# for
		elif item['Credibility'] in ['false', 'mostly false']: 
			articleLabels.append(1) # against

try:
	assert(len(claims) == len(labels))
except AssertionError as e:
	print ('length of claims: ' + str(len(claims)) + ' but length of labels: ' + str(len(labels)))


# lingustic classification
'''
if not (os.path.isfile('lgX.npy')):
	lgClaims = extractLGfeatures(claims, lgFeaturesPath)
	evaluateLG(lgClaims, labels)
else:
	evaluateLG(None, labels, isFeatureGenerated = True)
'''

'''
# stance classification
if not (os.path.isfile('relatedSnippetX.npy') 
		and os.path.isfile('relatedSnippet_y.npy')):
	evaluateStance(claims, articles, articleLabels)
else:
	evaluateStance(None, None, None, isFeatureGenerated = True)
'''

# stance classification with linguistic features
if not (os.path.isfile('relatedSnippetLgX.npy') 
		and os.path.isfile('relatedSnippetLg_y.npy')):
	evaluateStanceLg(claims, articles, articleLabels, lgFeaturesPath)
else:
	evaluateStanceLg(None, None, None, lgFeaturesPath, isFeatureGenerated = True)

'''
how to concat features since some claims do not have quotes
doesn't matter because stance feature is the stance classifier's score on the articles from the web, which is trained from corpus of Snopes "descriptions."
'''