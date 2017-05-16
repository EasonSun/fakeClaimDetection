import numpy as np
import os
import re
import sys
import time
import io

from Classifier import Classifier
from StanceReader import StanceReader
from relatedSnippetsExtractor import relatedSnippetsExtractor
from lgExtractor import lgExtractor


experimentPath = sys.argv[1]

relatedSnippetsPath = experimentPath + 'stance/relatedSnippets.txt'
relatedSnippetLabelsPath = experimentPath + 'stance/relatedSnippetLabel'
claimXPath = experimentPath + 'stance/claimX'
relatedSnippetsXPath = experimentPath + 'stance/relatedSnippetX'


featureNamePath = experimentPath + 'stance/featureName'
wrongPredicitonPath = experimentPath + 'stance/wrongPrediciton.txt'
logPath = experimentPath + 'log.txt'

MIN_DF = float(sys.argv[2])
MAX_DF = float(sys.argv[3])
overlapThreshold = float(sys.argv[4])

snopeDataPath = sys.argv[5]
doc2vecPath = sys.argv[6]

lgPath = "data/linguisticFeatures/allFeatures.txt"

reader = StanceReader(snopeDataPath)
rsExtractor = relatedSnippetsExtractor(overlapThreshold, doc2vecPath=doc2vecPath)

# no need of parellel
def read(relatedSnippets, relatedSnippetLabels, claimX, relatedSnippetsX):
	for filePath in os.listdir(snopeDataPath):
		if not filePath.endswith('.json'):
			continue
		claim, article, cred = reader.readSnopes(filePath)
		if (claim == 'the name of the san diego wild animal parks monorail was taken from crude acronym'):
			print filePath
		if claim is None:
			continue
		#t1 = time.clock()
		claimX_, relatedSnippetsX_, relatedSnippets_, relatedSnippetLabels_ = rsExtractor.extract(claim, article, label=cred)
		#t2 = time.clock()
		#print (t2-t1)
		if relatedSnippets_ is not None:
			if claimX is None:
				claimX = claimX_
				relatedSnippetsX = relatedSnippetsX_
			else:
				np.vstack((claimX, claimX_))
				try:
					np.vstack((relatedSnippetsX, relatedSnippetsX_))
				except ValueError:
					print (relatedSnippetsX_.shape, relatedSnippetsX.shape)
					print (filePath)
					break
			relatedSnippets.extend(relatedSnippets_)
			relatedSnippetLabels.extend(relatedSnippetLabels_)
		
	print (relatedSnippetsX.shape)

	with io.open(relatedSnippetsPath, 'w') as f:
		for item in relatedSnippets:
			f.write(item + '\n')
	np.save(claimXPath, claimX)
	np.save(relatedSnippetsXPath, relatedSnippetsX)
	relatedSnippetLabels = np.array(relatedSnippetLabels)
	np.save(relatedSnippetLabels, relatedSnippetLabelsPath)

	print ('finish related snippets extraction')
	ratioImbalance = np.sum(relatedSnippetLabels) / (relatedSnippetLabels.shape - np.sum(relatedSnippetLabels))
	print ('ratio of imbalance, neg : pos is %4f' %ratioImbalance)
	print("MIN_DF = %f" %MIN_DF)
	print("MAX_DF = %f" %MAX_DF)

	logFile = io.open(logPath, 'a')
	logFile.write ('ratio of imbalance, neg : pos is  %4f \n' %ratioImbalance)
	logFile.write("MIN_DF = %f \n" %MIN_DF)
	logFile.write("MAX_DF = %f \n" %MAX_DF)
	logFile.close()


	
def main():
	print ('start stance evaluation ...')
	'''
	claims = []
	articles = []
	creds = []
	'''
	relatedSnippets = []
	relatedSnippetLabels = []
	claimX = None
	relatedSnippetsX = None

	if not os.path.isfile(relatedSnippetsXPath+'.npy'):
		read(relatedSnippets, relatedSnippetLabels, claimX, relatedSnippetsX)
		# relatedSnippetLabels = np.array(relatedSnippetLabels)
	else:
		with io.open(relatedSnippetsPath) as f:
			relatedSnippets = f.readlines()
			print(relatedSnippets[0])
		claimX = np.load(claimXPath + '.npy')
		relatedSnippetsX = np.load(relatedSnippetsXPath + '.npy')
		relatedSnippetLabels = np.load(relatedSnippetLabelsPath + '.npy')

	relatedSnippet_y = np.array(0)
	featureNames = []
	'''
	if not os.path.isfile(relatedSnippetsXPath+'.npy'):
		
		relatedSnippetsX, featureNames = extractor.extractFeatures(relatedSnippets, MIN_DF, MAX_DF)
		relatedSnippet_y = np.array(relatedSnippetLabels)
		np.save(relatedSnippetsXPath, relatedSnippetsX)
		np.save(relatedSnippet_yPath, relatedSnippet_y)
		np.save(featureNamePath, np.array(featureNames))
		
		LGExtractor = lgExtractor(lgPath)
		relatedSnippetsX, featureNames = LGExtractor.extract(relatedSnippets)
		relatedSnippet_y = np.array(relatedSnippetLabels)
		np.save(relatedSnippetsXPath, relatedSnippetsX)
		np.save(relatedSnippet_yPath, relatedSnippet_y)
		np.save(featureNamePath, featureNames)

	else:
		try:
			relatedSnippetsX = np.load((relatedSnippetsXPath + '.npy')).item()
		except:
			relatedSnippetsX = np.load((relatedSnippetsXPath + '.npy'))
		try:
			relatedSnippet_y = np.load(relatedSnippet_yPath + '.npy').item()
		except:
			relatedSnippet_y = np.load(relatedSnippet_yPath + '.npy')
		featureNames = np.load(featureNamePath+'.npy')
	'''
	print ('start classifying')
	clf = Classifier(relatedSnippetsX, 'stance', logPath, experimentPath, y=relatedSnippetLabels)
	
	# clf.evaluateFeatureImportance(featureNames, max_depth=30)
	clf.paramSearch()
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
	relatedSnippetsX = np.concatenate((relatedSnippetsX, relatedSnippetMarkNumberX), axis=1)
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt')
	from sklearn.model_selection import GridSearchCV
	grid = dict(n_estimators=[500], max_depth=[2000])
	forestGS = GridSearchCV(estimator=forest, param_grid=grid, cv=5)
	forestGS.fit(relatedSnippetsX, relatedSnippet_y)

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