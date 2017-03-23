#!/bin/bash
dataPath='../../Data/Snopes'
lgFeaturesPath='../../Data/linguisticFeatures/'

nfold='10'
python3 main.py $dataPath $lgFeaturesPath