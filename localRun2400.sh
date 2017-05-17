#!/bin/bash
snopeDataPath="data/Snopes"
googleDataPath="data/Google_processed"
glovePath="data/glove/glove.6B.200d.txt"
doc2vecPath='data/doc2vec_apnews.bin'
lgPath="data/linguisticFeatures/allFeatures.txt"
resultPath="results/"
experimentPath="results/experiment_$1/"
txt="txt"
### experimentLogPath="$experimentLogPath.$txt"	###??? doesn't work?

###	intialize experiment
if [ ! -e $experimentPath ]
then
	mkdir $experimentPath
	mkdir "experimentPath/stance"
fi

overlapThreshold=".44"
# tweek the following parameters when you have related snippets
MIN_DF=".006"
MAX_DF=".5"
echo overlapThreshold >> experimentLogPath
### read data
### python3 src/readData.py $snopeDataPath $experimentPath $gloveDataPath
### evaluate stance classifier
### python src/evaluateStance.py $experimentPath $MIN_DF $MAX_DF $overlapThreshold $snopeDataPath $doc2vecPath
### evaluate claim credibility
overlapThreshold=".55"
python src/evaluateClaim.py $experimentPath $MIN_DF $MAX_DF $overlapThreshold $lgPath $snopeDataPath $googleDataPath $doc2vecPath 2001 2400