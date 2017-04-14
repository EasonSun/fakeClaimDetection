import numpy as np
import os
import re
import sys
import time
from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor
from lgExtractor import lgExtractor


experimentPath = sys.argv[1]
claimPath = experimentPath + 'claims.txt'
articlePath = experimentPath + 'articles.txt'
stancePath = experimentPath + 'stance.npy'
relatedSnippetsPath = experimentPath + 'stance/relatedSnippets.txt'
relatedSnippetLabelsPath = experimentPath + 'stance/relatedSnippetLabels.npy'

relatedSnippetXPath = experimentPath + 'stance/relatedSnippetX'
relatedSnippet_yPath = experimentPath + 'stance/relatedSnippet_y'
featureNamePath = experimentPath + 'stance/featureName'
wrongPredicitonPath = experimentPath + 'stance/wrongPrediciton.txt'
logPath = experimentPath + 'log.txt'

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

lgPath = "data/linguisticFeatures/allFeatures.txt"

	
def main():
	print ('start stance evaluation ...')
	claims = []
	with open(claimPath) as f:
	    claims = f.readlines()
	claims = [x.strip() for x in claims] 

	articles = []
	with open(articlePath) as f:
	    articles = f.readlines()
	articles = [x.strip() for x in articles]
	articleLabels = np.load(stancePath)

	lgFeatures = {}
	nextValue = 0
	with open(lgPath) as f:
		for lgFeature in f:
			lgFeature = lgFeature.rstrip()
			if lgFeature not in lgFeatures:
				lgFeatures[lgFeature] = nextValue
				nextValue += 1

	relatedSnippets = []
	relatedSnippetLabels = []

	extractor = relatedSnippetsExtractor(overlapThreshold)

	if not os.path.isfile(relatedSnippetsPath):
		for claim, article, articleLabel in zip(claims, articles, articleLabels):
			relatedSnippets_, relatedSnippetLabels_ = extractor.extract(claim, article, articleLabel)
			if relatedSnippets_ is not None:
				relatedSnippets.extend(relatedSnippets_)
				relatedSnippetLabels.extend(relatedSnippetLabels_)

		with open(relatedSnippetsPath, 'w') as f:
			for item in relatedSnippets:
				f.write(item + '\n')
		np.save(relatedSnippetLabelsPath, np.array(relatedSnippetLabels))

		# can be renamed to stanceRelatedSnippetsPath
		# that means save two versions of RelatedSnippets, one for stance one for 	claim classification
		# but maybe no need to do that

	else:
		with open(relatedSnippetsPath) as f:
		    relatedSnippets = f.readlines()
		relatedSnippets = [x.strip() for x in relatedSnippets]
		relatedSnippetLabels = np.load(relatedSnippetLabelsPath)
		#the above line does not make sense?
	print ('finish related snippets extraction')
	ratioImbalance = np.sum(relatedSnippetLabels) / (relatedSnippetLabels.shape - np.sum(relatedSnippetLabels))
	print ('ratio of imbalance, neg : pos is %4f' %ratioImbalance)
	print("MIN_DF = %f" %MIN_DF)
	print("MAX_DF = %f" %MAX_DF)

	logFile = open(logPath, 'a')
	logFile.write ('ratio of imbalance, neg : pos is  %4f \n' %ratioImbalance)
	logFile.write("MIN_DF = %f \n" %MIN_DF)
	logFile.write("MAX_DF = %f \n" %MAX_DF)
	logFile.close()


	relatedSnippetX = np.array(0)
	relatedSnippet_y = np.array(0)
	featureNames = []

	if not os.path.isfile(relatedSnippetXPath+'.npy'):
		'''
		relatedSnippetX, featureNames = extractor.extractFeatures(relatedSnippets, MIN_DF, MAX_DF)
		relatedSnippet_y = np.array(relatedSnippetLabels)
		np.save(relatedSnippetXPath, relatedSnippetX)
		np.save(relatedSnippet_yPath, relatedSnippet_y)
		np.save(featureNamePath, np.array(featureNames))
		'''
		LGExtractor = lgExtractor(lgFeatures)
		relatedSnippetX, featureNames = LGExtractor.extract(relatedSnippets)
		relatedSnippet_y = np.array(relatedSnippetLabels)
		np.save(relatedSnippetXPath, relatedSnippetX)
		np.save(relatedSnippet_yPath, relatedSnippet_y)
		np.save(featureNamePath, featureNames)

	else:
		try:
			relatedSnippetX = np.load((relatedSnippetXPath + '.npy')).item()
		except:
			relatedSnippetX = np.load((relatedSnippetXPath + '.npy'))
		try:
			relatedSnippet_y = np.load(relatedSnippet_yPath + '.npy').item()
		except:
			relatedSnippet_y = np.load(relatedSnippet_yPath + '.npy')
		featureNames = np.load(featureNamePath+'.npy')

	print ('start classifying')
	clf = Classifier(relatedSnippetX, 'stance', logPath, experimentPath, relatedSnippet_y)
	
	clf.evaluateFeatureImportance(featureNames, max_depth=30)
	# clf.paramSearch()
	clf.crossValidate(max_depth=300, n_fold=10)
	# clf.crossValidate(max_depth=80)
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
	'''
	def evaluateStanceLg(claims, articles, articleLabels, lgFeaturesPath, isSnippetGenerated=False):
		relatedSnippetLgX = np.array(0)
		relatedSnippetLg_y = np.array(0)
		# relatedSnippetMarkNumberX = np.array(0)
		if not isSnippetGenerated:
			relatedSnippet, relatedSnippetLabels = extractRelatedSnippets(claims, articles, articleLabels)
			relatedSnippetLg = extractLGfeatures(relatedSnippet, lgFeaturesPath)
			# too many features!
			from sklearn.feature_extraction.text import CountVectorizer
			vectorizer = CountVectorizer(analyzer = "word",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             stop_words = None) # no need to strip off stop_words because all linguistic features
			
			relatedSnippetLgX = vectorizer.fit_transform(relatedSnippetLg)
			# print (vectorizer.vocabulary_)
			
			relatedSnippetLgX = relatedSnippetLgX.toarray()
			np.save('relatedSnippetLgX', relatedSnippetLgX)

			relatedSnippetLg_y = np.array(relatedSnippetLabels)
			np.save('relatedSnippetLg_y', relatedSnippetLg_y)

		else:
			relatedSnippetLgX = np.load('relatedSnippetLgX.npy')
			relatedSnippetLg_y = np.load('relatedSnippetLg_y.npy')
		
		Classifier(relatedSnippetLgX, relatedSnippetLg_y, vectorizer.featureNames)
	'''

if __name__ == '__main__':
    main()