import numpy as np
import os

def evaluateLG(lgClaims, y, isFeatureGenerated=False):
	#print (lgClaims)
	X = np.array(0)
	y = np.array(y)
	if not isFeatureGenerated:
		from sklearn.feature_extraction.text import CountVectorizer
		# empty string can be taken as all 0 vectors
		vectorizer = CountVectorizer(analyzer = "word",   \
		                             tokenizer = None,    \
		                             preprocessor = None, \
		                             #keep !, ?, ', and " as features
		                             token_pattern = '(?u)\\b\\w\\w+\\b|!|\\?|\\"|\\\'', \
		                             stop_words = None) # no need to strip off stop_words because all linguistic features
		                             #max_features = 5000) 

		X = vectorizer.fit_transform(lgClaims)
		X = X.toarray()
		np.save('lgX', X)
	else:
		X = np.load('lgX.npy')

	nunSample, numFeature = X.shape	
	print ('nunSample and numFeature of linguistic features')
	print (nunSample, numFeature)
	ratioImbalance = np.sum(y) / (y.shape - np.sum(y))
	print ('ratio of imbalance, neg : pos is %4f' %ratioImbalance)
	resultFile = open('lgResult.txt', 'w')	
	resultFile.write ('nunSample and numFeature of linguistic features')
	resultFile.write (nunSample, numFeature)
	resultFile.write ('ratio of imbalance, neg : pos is %4f' %ratioImbalance)
	'''
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	
	param_grid=dict(n_estimators=np.linspace(500, 100, 1100), max_depth=np.linspace(nunSample/2, 5, nunSample))

	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import classification_report
	from sklearn.svm import SVC
	'''
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier(max_features='sqrt')

	from sklearn.model_selection import GridSearchCV
	# grid search for the best parameter
	# grid = dict(n_estimators=range(500, 1100, 100), max_depth=range(int(nunSample/2), nunSample, 5))
	grid = dict(n_estimators=[500, 1000], max_depth=[3000, 4500])
	forestGS = GridSearchCV(estimator=forest, param_grid=grid, cv=5)
	forestGS.fit(X,y)

	means = forestGS.cv_results_['mean_test_score']
	stds = forestGS.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, forestGS.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
		resultFile.write ("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	
	'''
	0.691 (+/-0.017) for {'max_depth': 3000, 'n_estimators': 500}
	0.691 (+/-0.024) for {'max_depth': 3000, 'n_estimators': 1000}
	0.691 (+/-0.026) for {'max_depth': 4500, 'n_estimators': 500}
	0.690 (+/-0.023) for {'max_depth': 4500, 'n_estimators': 1000}
	'''