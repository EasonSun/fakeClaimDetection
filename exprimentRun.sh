#!/bin/bash
dataPath='data/Snopes'
lgFeaturesPath='data/linguisticFeatures/'
resultPath='results/'
srcPath='src/'
experimentPath='results/experiment $1/'
experimentLogPath='results/log.txt'


###	intialize experiment
mkdir experimentPath
touch experimentLogPath

cd srcPath
overlapThreshold='.09'
MIN_DF='.005'
MAX_DF='.38'
echo overlapThreshold >> experimentLogPath
### read data
python3 readData.py $$dataPath $experimentPath
### extract related snippets
python3 extractRelatedSnippets.py $experimentPath $overlapThreshold
### evaluate stance classifier
python3 evaluateStance.py $experimentPath $MIN_DF $MAX_DF $experimentLogPath