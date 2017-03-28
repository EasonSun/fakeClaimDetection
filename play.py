import spacy
nlp = spacy.load('en')

# s = u'Actor Jack Nicholson said that he was \"positively against\" abortion.'
#s = u'Missouri State University students protested to get Abraham Lincoln removed from all U.S. currency because they believe that the former president was racist.'

s = u'A female Air Force National Guard member was denied service at a Bloomington, Minnesota, SuperAmerica station because her military uniform offended foreign cab drivers.'
def cao(s):
	doc = nlp(s)
	for ent in doc.ents:
		print(ent.label_, ent.text)
		#print (ent.merge(ent.tag_, ent.text, ent.ent_type_))
	
	# same as collapse phrases as in the visualiser
	for np in list(doc.noun_chunks):
		np.merge(np.root.tag_, np.root.lemma_, np.root.ent_type_)
	'''
	https://spacy.io/docs/usage/entity-recognition
	https://spacy.io/docs/usage/dependency-parse
	
	how visualizer works:
	https://github.com/explosion/spacy-services/blob/master/displacy/displacy_service/parse.py

	dependencies list:
	https://github.com/explosion/spaCy/issues/233
	'''
	print([tok.dep_ for tok in doc])
	nsubj = [tok for tok in doc if (tok.dep_ in ["nsubjpass", "nsubj"]) ]
	print (nsubj)	#can't differentiate clauses?
	obj = [tok for tok in doc if (tok.dep_ in ["dobj", "pobj", "oprd", "iobj"]) ]
	print (obj)
	
cao(s)

