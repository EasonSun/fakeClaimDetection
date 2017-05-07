#!/bin/bash
snopeDataPath="../data/Snopes"
googleDataPath="../data/Google_processed"
lgPath="../data/linguisticFeatures/allFeatures.txt"
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
# python3 src/readData.py $snopeDataPath $experimentPath
### evaluate stance classifier
### python3 src/evaluateStance.py $experimentPath $MIN_DF $MAX_DF $overlapThreshold
### evaluate claim credibility
overlapThreshold=".02"
python3 src/evaluateClaim\ copy.py $experimentPath $MIN_DF $MAX_DF $overlapThreshold $lgPath $snopeDataPath $googleDataPath