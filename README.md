# fakeClaimDetection
## Files
src/\
⋅⋅⋅⋅StanceReader.py reader for reading data for stance classification\
⋅⋅⋅⋅ClaimReader.py reader for reading data for claim classification (final credibility assessment)\
⋅⋅⋅⋅relatedSnippetsExtractor.py class for extracting related snippets\
⋅⋅⋅⋅lgExtractor.py class for extracting linguistic features\
⋅⋅⋅⋅evaluateStance.py script for stance classification\
⋅⋅⋅⋅evaluateClaim.py script for claim classification (final credibility assessment)\
_You need to create a data/ folder and down the files_ (click on each name for its download page)\
The Google_processed folder is not for test purpose; you don't need to download it for testing.
data/\
⋅⋅⋅⋅[Snopes](https://www.dropbox.com/s/q37ilebwotc9hj3/Snopes.zip?dl=0)\
⋅⋅⋅⋅[Google_processed](https://www.dropbox.com/s/7mzrpah7csubagz/Google_processed.zip?dl=0)\
⋅⋅⋅⋅[Google_test](https://www.dropbox.com/s/t2if7r638dry3ma/Google_test.zip?dl=0)\
⋅⋅⋅⋅[doc2vec_apnews.bin](https://ibm.ent.box.com/s/9ebs3c759qqo1d8i7ed323i6shv2js7e)\
⋅⋅⋅⋅doc2vec_apnews.bin.syn0.npy\
⋅⋅⋅⋅doc2vec_apnews.bin.syn1neg.npy\
⋅⋅⋅⋅[stanceRF120.pkl](https://www.dropbox.com/s/xqd0h7a042nub2u/rf.pkl?dl=0)\
Thanks for jhlau's pretrained [doc2vec](https://github.com/jhlau/doc2vec)\

## Run
Create a virtualenv with python2 and no system package, like\
`mkvirtualenv -p python_path --no-system-package`\
Then run\
`pip install -r requirements.txt`\
Then install the specific version of gensim from [here](https://github.com/jhlau/gensim)\
To run, you need to specify a experiment name, which will be created as a relateive path inside "results/"\
`./localRun.sh $experimentPath`\
For the current purpose, functionalities other than evaluateClaim is disabled.\

## Example output
food_warnings_salad.json\
Prepackaged salads and spinach may contain E. coli.\
label: 0 predict: 0 confidence: 0.61462561862\
[u'share of leafy greens sales volume 2005 change in sales volume bagged salads without spinach', u'coli from spinach grown on single california field investigators traced the prepackaged spinach back to natural selection foods and baby spinach five deaths were linked to the outbreak', u'went out in september 2006 at least 205 reports of illnesses and three deaths across twentyfive states were confirmed to have been caused by e all contaminated spinach was sold under the dole brand', u'in recent years there have been e coli outbreaks caused by contaminated signs symptoms and treatment', u'in summer 2010 more than , 900 people were reportedly sickened by salmonella found in eggs produced by which voluntarily recalled about halfbillion eggs nationwide', u'coli outbreak occurred in 2012 thirtythree people became sick and thirteen were hospitalized after eating two people suffered kidney failure', u'isolated from an opened package of baby spinach best if used by august 30 packed by in the refrigerator of an ill new mexico resident matched that of the outbreak strain', u'28563400000 artichoke wheatberry salad 28563700000 southwest soofoo salad 28563800000 southwest soofoo salad', u'as we reported in april there was an e coli outbreak last year linked to bag lettuce this time its allegedly bagged spinach that is contaminated', u'it was distributed throughout colorado kansas and missouri and through retail grocer dole recalls limited number of salads the salads may be contaminated with listeria monocytogenes']
