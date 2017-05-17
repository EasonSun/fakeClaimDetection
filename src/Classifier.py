import numpy as np
import os 
from sklearn.ensemble import RandomForestClassifier
import io

class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self, task, logPath, experimentPath, X=None, y=None):
		# X, y are numpy arrays to evaluate
		self.X = X
		self.y = y
		self.logPath = logPath
		self.experimentPath = experimentPath
		self.task = task
		self.rf = self._initRF()
		self._printClass()

	def _printClass(self):
		if self.X is None:
			print (self.rf.max_depth, self.rf.n_estimators) 
			return
		nunSample, numFeature = self.X.shape	
		print("nunSample, numFeature: %i, %i" %(nunSample, numFeature))
		logFile = io.open(self.logPath, 'a')
		logFile.write(u"nunSample, numFeature: %i, %i\n" %(nunSample, numFeature))
		logFile.close()

	def _initRF(self):
		modelPath = self.experimentPath+self.task+'/rf.pkl'
		if os.path.isfile(modelPath):
			print ('stance classifier loaded with max_depth and n_estimators:')
			from sklearn.externals import joblib
			return joblib.load(modelPath)
		else:
			if self.task == 'stance':
				return RandomForestClassifier(max_features='sqrt', class_weight='balanced', n_jobs=2, n_estimators=1000, max_depth=80)
			else:
				return RandomForestClassifier(max_features='sqrt', class_weight='balanced', n_jobs=2)

	def evaluateFeatureImportance(self, featureNames, n_fold=5, max_depth=None):
		importancesPath = self.experimentPath+self.task+'/importances'
		topFeaturesPath = self.experimentPath+self.task+'/topFeatures.txt'
		if max_depth is not None:
			self.rf.max_depth = max_depth
		importances = np.zeros(self.X.shape[1])
		# importancesPerClass = np.zeros((numFeature, 2))
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=n_fold)
		if not os.path.isfile(importancesPath+'.npy'):
			for train_index, test_index in kf.split(self.X):
				# print("TRAIN:", train_index, "TEST:", test_index)
				X_train, X_test = self.X[train_index], self.X[test_index]
				y_train, y_test = self.y[train_index], self.y[test_index]
				self.rf.fit(X_train, y_train)
				importances += self.rf.feature_importances_

			importances /= n_fold
			np.save(importancesPath, np.array(importances))	#should not change with param?
		else:
			importances = np.load(importancesPath+'.npy')
			# importancesPerClass = np.load('importancesPerClass.npy')

		assert (importances.shape[0] == len(featureNames))
		top10PercentIdx = np.argsort(importances)#[-int(numFeature*.1):]
		topFeatureFile = io.open(topFeaturesPath, 'w')
		for id in top10PercentIdx:
			topFeatureFile.write(str(featureNames[id])+'\t'+str(importances[id])+'\n')
		topFeatureFile.close()


	def paramSearch(self, n_fold=5, sampleWeight=None):
		logFile = io.open(self.logPath, 'a')
		print ('Start searching best parameters ...')
		logFile.write(u'Start searching best parameters ... \n')
		from sklearn.model_selection import GridSearchCV
		grid = dict(max_depth=[self.rf.max_depth*(1+i) for i in np.arange(0,1,.5)])
		# grid = dict(n_estimators=[500], max_depth=[2000])
		rfGS = GridSearchCV(estimator=self.rf, param_grid=grid, cv=n_fold, n_jobs=2)
		if (sampleWeight is not None):
			rfGS.fit_params={'sample_weight': sampleWeight}
		rfGS.fit(self.X, self.y)

		means = rfGS.cv_results_['mean_test_score']
		stds = rfGS.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, rfGS.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
			logFile.write(u"%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
		self.rf = rfGS.best_estimator_
		from sklearn.externals import joblib
		joblib.dump(rfGS.best_estimator_, self.experimentPath+self.task+'/rf.pkl')
		logFile.close()
		# sample_weight at fit is to deal with class imbalance

	# this is the real cross validation, after got the best params
	# X: (numArticle, numClass), y: (numArticle)
	# sourceCredByStance is source cred by stance (numArticle, numClass)
	# y_pred_prob: (numArticle, numClass)
	# if claimArticleIdx is not None, then calculate per claim y_pred


	def crossValidate(self, n_fold=5, sourceCredByStance=None, claimArticleIdx=None, max_depth=None):
		logFile = io.open(self.logPath, 'a')

		if (claimArticleIdx is None):
			logFile.write(u'Per article evaluation\n')
		else:
			logFile.write(u'Per claim evaluation\n')

		if max_depth is not None:
			self.rf.max_depth = max_depth

		from sklearn.metrics import classification_report
		from sklearn.metrics import accuracy_score
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=n_fold)
		for train_index, test_index in kf.split(self.X):
			X_train, X_test = self.X[train_index], self.X[test_index]
			y_train, y_test = self.y[train_index], self.y[test_index]
			self.rf.fit(X_train, y_train)
			y_pred_prob = self.rf.predict_log_proba(X_test)
			if sourceCredByStance is not None:
				sourceCred = sourceCredByStance[test_index]
				assert(y_pred_prob.shape == sourceCred.shape)
				y_pred_prob = np.multiply(y_pred_prob, sourceCred)
				if (claimArticleIdx is not None):
					# make it per claim, group by article
					_, idx, counts = np.unique(claimArticleIdx, return_inverse=True, return_counts=True)
					for i in range(sourceCred.shape[1]):
						y_pred_prob[i,:] = np.bincount(idx, weights=y_pred_prob[:,i]) / counts
					y_test = np.bincount(idx) / counts
			y_pred = np.argmax(y_pred_prob, axis=1)
			assert(y_pred.shape == y_test.shape)
			result = classification_report(y_test, y_pred, target_names=['true', 'fake'])
			accuracy = accuracy_score(y_test, y_pred)
			print(accuracy)
			#logFile.write(str(accuracy)+'\n')
			print(result)
			#logFile.write(result+'\n')
		logFile.close()


	# [n_samples, n_classes]
	def predict_porb(self, X=None):
		y_pred_prob = np.zeros(0)
		if (X is None):
			# rf has been refitted to the entire dataset after CV
			#y_pred_prob = self.rf.predict_log_proba(self.X)
			y_pred_prob = self.rf.predict_proba(self.X)
		else:
			#y_pred_prob = self.rf.predict_log_proba(X)
			y_pred_prob = self.rf.predict_proba(X)
		# np.save(self.experimentPath+task+'/stance_prob', y_pred_prob)
		
		return y_pred_prob

		'''
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=n_fold)
		y_pred_prob = np,zeros(self.X.shape[1])
		for train_index, test_index in kf.split(self.X):
			# print("TRAIN:", train_index, "TEST:", test_index)
			_, X_test = self.X[train_index], self.X[test_index]
			y_pred_prob += self.rf.predict_proba(X_test)
		y_pred_prob /= n_fold
		np.save(self.experimentPath+'', y_pred_prob)
		return y_pred_prob
		'''
	'''
	def weightedparamSearch(sourceCredn_fold=5):
		logFile = io.open(self.logPath, 'w')
		from sklearn.model_selection import GridSearchCV
		grid = dict(max_depth=[80, 90, 100])
		# grid = dict(n_estimators=[500], max_depth=[2000])
		rfGS = GridSearchCV(estimator=self.rf, param_grid=grid, cv=n_fold, n_jobs=2)
		rfGS.fit(self.X, self.y)

		means = rfGS.cv_results_['mean_test_score']
		stds = rfGS.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, rfGS.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
			logFile.write("%0.3f (+/-%0.03f) for %r \n" % (mean, std * 2, params))
		self.rf = rfGS.best_estimator_
		from sklearn.externals import joblib
		joblib.dump(rfGS.best_estimator_, self.experimentPath + 'rf.pkl')
		logFile.close()
	'''

	def predict(self, origDataPath=None):
		# print(rfGS.best_params_)
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
		self.rf.fit(X_train, y_train)
		y_pred = self.rf.predict(X_test)

		from sklearn.metrics import classification_report
		result = classification_report(y_test, y_pred, target_names=['true', 'fake'])
		print(result)
		logFile = io.open(self.logPath, 'a')
		logFile.write(result)
		logFile.close()

		if origDataPath == None:
			return

		wrongPrediction = []
		correctPrediction = []
		with io.open(origDataPath) as f:
			origData = f.readlines()
			origData = np.array([x.strip() for x in origData])
			wrongPrediction = origData[y_pred != y_test]
		with io.open(self.experimentPath+'wrongPrediciton.txt', 'w') as f:
			f.write('snippet'+'\t'+'prediciton'+'\t'+'label')
			l = len(wrongPrediction)
			for i in range(l):
				f.write(wrongPrediction[i]+'\t'+y_pred[i]+'\t'+y_test[i])
	
	'''
	Why training report is like this? 50 done, 150 done, and so on.
	[Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:   12.7s
	[Parallel(n_jobs=2)]: Done 196 tasks      | elapsed:   52.3s
	[Parallel(n_jobs=2)]: Done 446 tasks      | elapsed:  2.0min
	[Parallel(n_jobs=2)]: Done 796 tasks      | elapsed:  5.0min

	[Parallel(n_jobs=2)]: Done 1246 tasks      | elapsed:  7.9min
	[Parallel(n_jobs=2)]: Done 1796 tasks      | elapsed: 10.9min
	[Parallel(n_jobs=2)]: Done 2000 out of 2000 | elapsed: 11.8min finished
	'''


	'''
	http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
	This is how sklearn rf does scoring / loss function
	We can customize this if we use cred as some sort of weights
	but this will be used also in the training phase.
	'''