#!/bin/bash
dataPath="data/Snopes"
lgPath="data/linguisticFeatures/allFeatures.txt"
articlePath="data/articles.json"
resultPath="results/"
srcPath="src/"
experimentPath="results/experiment_$1/"
txt="txt"
### experimentLogPath="$experimentLogPath.$txt"	###??? doesn't work?

###	intialize experiment
if [ ! -e $experimentPath ]
then
	mkdir $experimentPath
	touch $experimentLogPath
fi

overlapThreshold=".04"
# tweek the following parameters when you have related snippets
MIN_DF=".006"
MAX_DF=".5"
echo overlapThreshold >> experimentLogPath
### read data
python3 src/readData.py $dataPath $experimentPath
### evaluate stance classifier
python3 src/evaluateStance.py $experimentPath $MIN_DF $MAX_DF $overlapThreshold
### evaluate claim credibility
###python3 src/evaluateClaim.py $experimentPath $MIN_DF $MAX_DF $overlapThreshold $lgPath $articlePath