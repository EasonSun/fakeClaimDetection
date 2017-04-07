import numpy as np
import os
import re
from extractLGfeatures import extractLGfeatures
from extractRelatedSnippets import extractRelatedSnippets

# hyper parameter

experimentPath = sys.argv[1]
claimPath = experimentPath + 'claims.txt'
articlePath = experimentPath + 'articles.txt'
credPath = experimentPath + 'cred.npy'
relatedSnippetsPath = experimentPath + 'relatedSnippets.txt'
relatedSnippetLabelsPath = experimentPath + 'relatedSnippetLabels.npy'

relatedSnippetXPath = experimentPath + 'relatedSnippetX'
relatedSnippet_yPath = experimentPath + 'relatedSnippet_y'
featureNamePath = experimentPath + 'featureName'
importancesPath = experimentPath + 'importances'
topFeaturesPath = experimentPath + 'topFeatures.txt'

MIN_DF = int(sys.argv[2])
MAX_DF = int(sys.argv[3])
logPath = sys.argv[4]

def evaluateHelper(X, y, featureNames):
	print (X.shape)
	nunSample, numFeature = X.shape	
	print("nunSample, numFeature: ")
	print (nunSample, numFeature)	#3227, 2929
	ratioImbalance = np.sum(y) / (y.shape - np.sum(y))
	print ('ratio of imbalance, neg : pos  is %4f' %ratioImbalance)
	print ("overlapThreshold = %f" %overlapThreshold)
	print("MIN_DF = %f" %MIN_DF)

	logFile = open(logPath, 'w')
	logFile.write("X dim and y dim: \n")
	nunSample, numFeature = X.shape	
	logFile.write('%i, %i \n' %(nunSample, numFeature))
	logFile.write ('ratio of imbalance, positive : negative is %4f \n' %ratioImbalance)
	logFile.write ("overlapThreshold = %f \n" %overlapThreshold)
	logFile.write("MIN_DF = %f \n" %MIN_DF)

	from sklearn import preprocessing
	X = preprocessing.scale(X) 
	
	'''
	# use LogisticRegression
	from sklearn.linear_model import LogisticRegression
	clf_l1_LR = LogisticRegression(C=50, penalty='l1', tol=0.0001)
	# http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers
	importances = np.zeros((1, numFeature))
	from sklearn.model_selection import KFold
	kf = KFold(n_splits=5)

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf_l1_LR.fit(X_train, y_train)
		importances += clf_l1_LR.coef_
	importances /= 5
	importances = importances[0]
	np.save('importancesPerClassLR', np.array(importances))
	top10PercentIdx = np.argsort(importances)#[-int(numFeature*.1):]
	
	# importances = np.load('importancesPerClassLR.npy')

	top10PercentIdx = np.argsort(importances)#[-int(numFeature*.1):]
	topFeatureFile = open('topFeaturesPerClassLR.txt', 'w')
	for id in top10PercentIdx:
		topFeatureFile.write(str(featureNames[id])+'\t'+str(importances[id])+'\n')
	'''

	
	# use RandomForestClassifier
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt', class_weight='balanced', n_jobs=1, n_estimators=5000, max_depth=1000)

	importances = np.zeros(numFeature)
	# importancesPerClass = np.zeros((numFeature, 2))
	if not os.path.isfile(importancesPath):
		#or not os.path.isfile('importancesPerClass.npy'):
		from sklearn.model_selection import KFold
		from treeinterpreter import treeinterpreter as ti
		kf = KFold(n_splits=5)
		
		for train_index, test_index in kf.split(X):
			# print("TRAIN:", train_index, "TEST:", test_index)
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			forest.fit(X_train, y_train)
			importances += forest.feature_importances_
			
			# importancesPerClassPerSplit = np.zeros((numFeature, 2))
			for x_test in X_test:
				x_test = x_test.reshape(x_test.shape[0], -1).T
				#print (x_test.shape)
				_, _, contributions = ti.predict(forest, x_test)
				# importancesPerClassPerSplit += contributions[0]
			# importancesPerClassPerSplit /= X_test.shape[0]
			# importancesPerClass += importancesPerClassPerSplit
			

		importances /= 5
		np.save(importancesPath, np.array(importances))	#not change with param.
		# importancesPerClass /= 5
		# np.save('importancesPerClass', np.array(importancesPerClass))

	else:
		importances = np.load(importancesPath + '.npy')
		# importancesPerClass = np.load('importancesPerClass.npy')

	top10PercentIdx = np.argsort(importances)#[-int(numFeature*.1):]

	# top10Features = [featureNames[id] f]
	topFeatureFile = open(topFeaturesPath, 'w')
	for id in top10PercentIdx:
		topFeatureFile.write(str(featureNames[id])+'\t'+str(importances[id])+'\n')

	'''
	# top feature per class by averaging testing on explore sets does not work
	topIndexPerClass = np.argsort(importancesPerClass, axis=0).T

	topFeatureFilePerClass = open('topFeaturesPerClass.txt', 'w')
	for topIndex in topIndexPerClass:
		for id in topIndex:
			topFeatureFilePerClass.write(str(featureNames[id])+'\t'+str(importances[id])+'\n')
		topFeatureFilePerClass.write('==============================\n')
	'''

	# needs to split training and testing 
	
	'''
	# CV
	# write as a seperate file with input args indicating evaluate or predict
	from sklearn.model_selection import GridSearchCV
	grid = dict(n_estimators=[2000, 4000], max_depth=[500, 1000])
	# grid = dict(n_estimators=[500], max_depth=[2000])
	forestGS = GridSearchCV(estimator=forest, param_grid=grid, cv=5, n_jobs=1, refit=False, verbose=10)	#no need to refit the entire dataset for prediction
	forestGS.fit(X, y)

	means = forestGS.cv_results_['mean_test_score']
	stds = forestGS.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, forestGS.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		logFile.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
	'''
	

if os.path.isfile(topFeaturesPath):
	return

relatedSnippetsFile = open(relatedSnippetsPath)
relatedSnippets = []
with open(relatedSnippetsFile) as f:
    relatedSnippets = f.readlines()
relatedSnippets = [x.strip() for x in relatedSnippets]

relatedSnippetLabels = np.load(relatedSnippetLabelsPath)

relatedSnippetX = np.array(0)
relatedSnippet_y = np.array(0)
featureNames = []

if not os.path.isfile(relatedSnippetLabelXPath):
	from sklearn.feature_extraction.text import TfidfVectorizer
	# empty string can be taken as all 0 vectors
	# using both uni- and bi-grams
	vectorizer = TfidfVectorizer(analyzer = "word", \
								 stop_words = "english",   \
	                             tokenizer = None,    \
	                             preprocessor = None, \
	                             min_df=MIN_DF, \
	                             max_df=MAX_DF, \
	                             ngram_range=(1, 2))	
	'''
	the min df above is really important as the first step for fieature engineering
	.005 means only keep features apper more than .005 portion of docs
	that is roughly 486 docs
	'''
	relatedSnippetX = vectorizer.fit_transform(relatedSnippets)
	# print (vectorizer.vocabulary_)
	relatedSnippetX = (relatedSnippetX.toarray())
	np.save(relatedSnippetXPath, relatedSnippetX)
	relatedSnippet_y = np.array(relatedSnippetLabels)
	np.save(relatedSnippet_yPath, relatedSnippet_y)
	featureNames = vectorizer.get_feature_names()
	np.save(featureNamePath, np.array(feature_names))


	#relatedSnippetMarkNumberX = np.array(relatedSnippetMarkNumbers)
	#np.save('relatedSnippetMarkNumberX', relatedSnippetMarkNumberX)

	# print("relatedSnippetX dim and relatedSnippet_y dim: ")
	# print(relatedSnippetX.shape, relatedSnippet_y.shape)

else:
	relatedSnippetX = np.load(relatedSnippetXPath + '.npy')
	relatedSnippet_y = np.load(relatedSnippet_yPath + '.npy')
	featureNames = np.load(featureNamePath + '.npy').tolist()

evaluateHelper(relatedSnippetX, relatedSnippet_y, featureNames)
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
	
	evaluateHelper(relatedSnippetLgX, relatedSnippetLg_y, vectorizer.feature_names)
'''