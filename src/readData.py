import sys
import os
import json
import numpy
from extractLGfeatures import extractLGfeatures
from evaluateLG import evaluateLG
from evaluateStance import evaluateStance, evaluateStanceLg

dataPath = sys.argv[1]
experimentPath = sys.argv[2]
claimPath = experimentPath + 'claims.txt'
articlePath = experimentPath + 'articles.txt'
credPath = experimentPath + 'cred'

if os.path.isfile(claimPath):
	return

for item in thelist:
  thefile.write("%s\n" % item)

data = []
for filePath in os.listdir(dataPath):
	if filePath.endswith('.json'):
		file = open(os.path.join(dataPath, filePath), 'r', encoding="utf-8")
		data.append(json.load(file))
		file.close()

claimFile = open(claimPath, 'w')
articleFile = open(articlePath, 'w')
cerdList = []

for item in data:	# each item is a dict
	if item['Description'] != '':
		claimFile.write("%s\n" % item['Claim'])
		articleFile.write('%s\n' % item['Description'])
		if item['Credibility'] in ['true', 'mostly true']: 
			cerdList.append(0)	# for
		elif item['Credibility'] in ['false', 'mostly false']: 
			cerdList.append(1) # against

claimFile.close()
articleFile.close()
np.save(credPath, np.array(cerdList))