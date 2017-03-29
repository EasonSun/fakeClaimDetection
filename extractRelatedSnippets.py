import numpy as np
import re
from overlap_lsi import overlap

def extractSnippets(article):
    snippets = []
    # snippetMarkNumbers = []	#list of list, inner list records number of ! ? ""
    NSS = 4 # number of a snippet in a sentence
    articleSentences = re.split(r'[.|!|?]', article)
    '''
    badStart = -1
    for i in len(articleSentences):
        if len(articleSentences[i] == 1) and i != 0 and len(articleSentences[i-1]) > 0:
            badStart = i
        if len(articleSentences[i] > 1 and badStart != -1):
            for j in range(badStart-1, i):
                articleSentences[i] += articleSentences[j]
            badStart = -1
    '''

    '''
    articleSentences = []
    articleSentencesWithMarks = []
    sentence = ''
    for c in article:
    	if c not in ['?', '!', '.']:
    		sentence += c
    	else:
    		articleSentences.append(sentence)
    		if c in ['?', '!']:
    			sentence += c
    		else:
    			sentence += ' '
    		articleSentencesWithMarks.append(sentence)
    		sentence = ''
    print (articleSentences)
    # print(articleSentences)
    '''
    # no stripped kinda needed: like $100, "bill", these are critical
    # but vocab from the vectorizer is without these 
    for i in range(len(articleSentences)):
        sentence = " ".join(re.findall("[a-zA-Z0-9]+", articleSentences[i]))
        articleSentences[i] =  sentence
    
    while '' in articleSentences:
        articleSentences.remove('')
    '''
    badWord = []
    for sentence in articleSentences:
        if len(sentence) == 1:
            badWord.append(sentence)
    badWords = "\.".join(badWord)
            #articleSentences.remove(sentence)
    '''

    
    if (len(articleSentences) < NSS):
        return [" ".join(articleSentences)]
    for i in range(len(articleSentences) - NSS + 1):
        temp = articleSentences[i : i+NSS]
        snippet = ""
        for j in range(NSS):
            if snippet == '':
                snippet += temp[j]
            else:
                snippet += ' ' + temp[j]
        snippets.append(snippet)
        '''
        snippetMarkNumber = [0,0,0]
        for j in range(NSS):
        	curSentence = articleSentencesWithMarks[i+j]
        	if '!' in curSentence:
        		snippetMarkNumber[0] = 1
        	if '?' in curSentence:
        		snippetMarkNumber[1] = 1
        	if '"' in curSentence:
        		snippetMarkNumber[2] = 1

        snippetMarkNumbers.append(snippetMarkNumber)
        '''
    return snippets

def extractVocab(claims, snippets, vectorizer):
    result = {}
    try:
        raw1 = vectorizer.fit(claims).vocabulary_
        raw2 = vectorizer.fit(snippets).vocabulary_
    except ValueError:
        return {}
    # print (raw2)
    nextValue = 0
    for key, value in raw1.items():
        if key not in result.keys():
            result[key] = nextValue
            nextValue += 1
            # print (key)
    for key, value in raw2.items():
        if key not in result.keys():
            result[key] = nextValue
            nextValue += 1
            # print (key)
    return result
    

def extractRelatedSnippets(claims, articles, articleLabels, overlapThreshold):
    relatedSnippet = []
    relatedSnippetLabels = []
    #relatedSnippetMarkNumbers = []  # record number of ! ? "", list of list

    from sklearn.feature_extraction.text import CountVectorizer
    # empty string can be taken as all 0 vectors
    # using both uni- and bi-grams
    vectorizer = CountVectorizer(analyzer = "word", \
                                preprocessor = None, \
                                stop_words = 'english', \
                                ngram_range=(1, 2))
                                 #max_features = 5000) 

    #from sklearn.metrics.pairwise import cosine_similarity

    # claim: str, article: str, articleLabel: int
    for claim, article, articleLabel in zip(claims, articles, articleLabels):
        # print (article)
        # print (claim)
        snippets = extractSnippets(article)
        # print (snippets)
        '''
        # you need to save by concat
        sims = np(0)
        if os.path.isfile('ldaSimScore.npy'):
            sims = np.load('ldaSimScore.npy')
        else:
            _, sims = overlap(snippets, claim)
            np.save('ldaSimScore.npy', sims)
        '''
        _, sims = overlap(snippets, claim)
        # print (sims)

        overlapIdx = np.where(sims > overlapThreshold)[0]
        #print (overlapIdx)
        relatedSnippetLabels.extend([articleLabel for i in range(len(overlapIdx))])
        snippets = np.array([[snippet] for snippet in snippets])
        #print (snippets.shape)
        # from vector back to sentence to later use them in the same feature space
        relatedSnippet.extend([''.join(snippet) for snippet in snippets[overlapIdx].tolist()])
        # relatedSnippetMarkNumbers.extend([snippetMarkNumbers[i] for i in overlapIdx])
        # print(relatedSnippet)
        #print(relatedSnippetLabels)
        
        
        '''
        # find vocab for this pair so as to do vector similarity
        vectorizer.vocabulary = None
        vocab = extractVocab([claim], snippets, vectorizer)
        if len(vocab.keys()) == 0:
            # bad thing can happen
            continue 
        # print (vocab)
        vectorizer.vocabulary = vocab
        assert(vectorizer.vocabulary == vocab)
        claimX = vectorizer.fit_transform([claim])
        assert(vectorizer.vocabulary == vocab)
        claimX = claimX.toarray()
        # print(claimX.shape)
        # print(claimX[0][210])
        snippetsX = vectorizer.fit_transform(snippets)
        assert(vectorizer.vocabulary == vocab)
        snippetsX = snippetsX.toarray()
        # print(snippetsX.shape)

        similarityScore = cosine_similarity(claimX, snippetsX)
        if (np.count_nonzero(similarityScore) == 0):
            # bad and weird thing happens 
            continue
        minSimilarityScore = np.min(similarityScore[np.nonzero(similarityScore)])
        if (minSimilarityScore < overlapThreshold):
            continue
        # print (minSimilarityScore)
        overlapIdx = np.where(cosine_similarity(snippetsX, claimX) > overlapThreshold)[0]
        #print (overlapIdx)
        relatedSnippetLabels.extend([articleLabel for i in range(len(overlapIdx))])
        snippets = np.array([[snippet] for snippet in snippets])
        #print (snippets.shape)
        # from vector back to sentence to later use them in the same feature space
        relatedSnippet.extend([''.join(snippet) for snippet in snippets[overlapIdx].tolist()])
        # relatedSnippetMarkNumbers.extend([snippetMarkNumbers[i] for i in overlapIdx])
        #print(relatedSnippet)
        #print(relatedSnippetLabels)
        '''

        '''
        # printStat
        print (claim)
        print(claimX.shape)
        print(snippetsX.shape)
        print (minSimilarityScore)
        '''
    return relatedSnippet, relatedSnippetLabels#, relatedSnippetMarkNumbers