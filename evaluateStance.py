import numpy as np
import os
import re
from extractLGfeatures import extractLGfeatures
from extractRelatedSnippets import extractRelatedSnippets

# hyper parameter
MIN_DF = .005
overlapThreshold = .035

def evaluateHelper(X, y, feature_names=None):
	nunSample, numFeature = X.shape	
	print("nunSample, numFeature: ")
	print (nunSample, numFeature)	#3227, 2929
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

	#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.get_params
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt', class_weight='balanced', n_jobs=2)


	from sklearn.model_selection import KFold
	kf = KFold(n_splits=5)
	importances = np.zeros(numFeature)
	for train_index, test_index in kf.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		forest.fit(X, y)
		importances += forest.feature_importances_
	importances /= 5
	top10PercentIdx = np.argsort(importances)#[-int(numFeature*.1):]

	# top10Features = [feature_names[id] f]
	topFeatureFile = open('topFeatures.txt', 'w')
	for id in top10PercentIdx:
		topFeatureFile.write(str(feature_names[id])+'\t'+str(importances[id])+'\n')
	# needs to split training and testing 

	'''
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
	'''
	
def evaluateStance(claims, articles, articleLabels, isSnippetGenerated=False):
	relatedSnippetX = np.array(0)
	relatedSnippet_y = np.array(0)
	# relatedSnippetMarkNumberX = np.array(0)
	if not isSnippetGenerated:
		relatedSnippet, relatedSnippetLabels = extractRelatedSnippets(claims, articles, articleLabels, overlapThreshold)
		from sklearn.feature_extraction.text import CountVectorizer
		# empty string can be taken as all 0 vectors
		# using both uni- and bi-grams
		vectorizer = CountVectorizer(analyzer = "word", \
									 stop_words = "english",   \
		                             tokenizer = None,    \
		                             preprocessor = None, \
		                             min_df=MIN_DF, \
		                             ngram_range=(1, 2))	
		'''
		the min df above is really important as the first step for fieature engineering
		.005 means only keep features apper more than .005 portion of docs
		that is roughly 486 docs
		'''
		relatedSnippetX = vectorizer.fit_transform(relatedSnippet)
		# print (vectorizer.vocabulary_)
		relatedSnippetX = (relatedSnippetX.toarray()).astype(float)
		from sklearn.feature_extraction.text import TfidfTransformer
		transformer = TfidfTransformer(smooth_idf=False)
		relatedSnippetX = transformer.fit_transform(relatedSnippetX)
		relatedSnippet_y = np.array(relatedSnippetLabels)

		np.save('relatedSnippetX', relatedSnippetX)
		np.save('relatedSnippet_y', relatedSnippet_y)
		evaluateHelper(relatedSnippetX, relatedSnippet_y, feature_names = vectorizer.get_feature_names())


		#relatedSnippetMarkNumberX = np.array(relatedSnippetMarkNumbers)
		#np.save('relatedSnippetMarkNumberX', relatedSnippetMarkNumberX)

		# print("relatedSnippetX dim and relatedSnippet_y dim: ")
		# print(relatedSnippetX.shape, relatedSnippet_y.shape)

	else:
		relatedSnippetX = np.load('relatedSnippetX.npy')
		relatedSnippet_y = np.load('relatedSnippet_y.npy')
		# relatedSnippetMarkNumberX = np.load('relatedSnippetMarkNumberX.npy')

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
                             #keep !, ?, ', and " as features
                             token_pattern = '(?u)\\b\\w\\w+\\b|!|\\?|\\"|\\\'', \
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
	
	evaluateHelper(relatedSnippetLgX, relatedSnippetLg_y, vectorizer.feature_names)
