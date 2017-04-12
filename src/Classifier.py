import numpy as np
import os 
from sklearn.ensemble import RandomForestClassifier


class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self, X, task, logPath, experimentPath, y=None):
		# X, y are numpy arrays to evaluate
		self.X = X
		self.y = y
		self.logPath = logPath
		self.experimentPath = experimentPath
		self.rf = self._initRF(task)
		self._printClass()

	def _printClass(self):
		nunSample, numFeature = self.X.shape	
		print("nunSample, numFeature: ")
		print (nunSample, numFeature)

		logFile = open(self.logPath, 'w')
		logFile.write("X dim and y dim: \n")
		nunSample, numFeature = self.X.shape	
		logFile.write('%i, %i \n' %(nunSample, numFeature))
		logFile.close()

	def _initRF(self, task):
		if os.path.isfile(self.experimentPath + task + 'rf.pkl'):
			from sklearn.externals import joblib
			return joblib.load(self.experimentPath + task + 'rf.pkl')
		else:
			if task == 'stance':
				return RandomForestClassifier(max_features='sqrt', class_weight='balanced', n_jobs=2, n_estimators=1000, max_depth=30)
			else:
				return RandomForestClassifier(max_features='sqrt', class_weight='balanced', n_jobs=2)

	def evaluateFeatureImportance(self, featureNames, n_fold=5):
		importancesPath = self.experimentPath + 'importances'
		topFeaturesPath = self.experimentPath + 'topFeatures.txt'

		importances = np.zeros(self.X.shape[1])
		# importancesPerClass = np.zeros((numFeature, 2))
		from sklearn.model_selection import KFold
		kf = KFold(n_splits=n_fold)
		if not os.path.isfile(importancesPath+'.npy'):
			#or not os.path.isfile('importancesPerClass.npy'):
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
		topFeatureFile = open(topFeaturesPath, 'w')
		for id in top10PercentIdx:
			topFeatureFile.write(str(featureNames[id])+'\t'+str(importances[id])+'\n')
		topFeatureFile.close()


	def crossValidate(self, n_fold=5):
		logFile = open(self.logPath, 'w')
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


	def predict_porb(n_fold=5):
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


	def predict(origDataPath=None):
		# print(rfGS.best_params_)
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
		self.rf.fit(X_train, y_train)
		y_pred = self.rf.predict(X_test)

		from sklearn.metrics import classification_report
		result = classification_report(y_test, y_pred, target_names=['true', 'fake'])
		print(result)
		logFile = open(self.logPath, 'w')
		logFile.write(result)
		logFile.close()

		if origDataPath == None:
			return

		wrongPrediction = []
		correctPrediction = []
		with open(origDataPath) as f:
			origData = f.readlines()
			origData = np.array([x.strip() for x in origData])
			wrongPrediction = origData[y_pred != y_test]
		with open(self.experimentPath+'wrongPrediciton.txt', 'w') as f:
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