#!/bin/bash
dataPath="data/Snopes"
lgFeaturesPath="data/linguisticFeatures/"
stopwordsPath="data/stopword.txt"
resultPath="results/"
srcPath="src/"
experimentPath="results/experiment_$1/"
experimentLogPath="$experimentPath_log"
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
### extract related snippets
python3 src/extractRelatedSnippets.py $experimentPath $overlapThreshold $stopwordsPath
### evaluate stance classifier
python3 src/evaluateStance.py $experimentPath $MIN_DF $MAX_DF $experimentLogPath
### evaluate claim credibility