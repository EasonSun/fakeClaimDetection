import numpy as np
import re
#from overlap_lsi import overlap
experimentPath = sys.argv[1]
claimPath = experimentPath + 'claims.txt'
articlePath = experimentPath + 'articles.txt'
credPath = experimentPath + 'cred.npy'
relatedSnippetsPath = os.path.join(experimentPath, 'relatedSnippets.txt')
relatedSnippetLabelsPath = os.path.join(experimentPath, 'relatedSnippetLabels')
overlapThreshold = int(sys.argv[2])

def extractSnippets(article):
    snippets = []
    # snippetMarkNumbers = []   #list of list, inner list records number of ! ? ""
    NSS = 4 # number of a snippet in a sentence
    articleSentences = re.split(r'[.|!|?]', article)
    
    '''
    # this tries to keep ! ? " marks
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
        sentence = " ".join(re.findall("[a-zA-Z0-9'-]+", articleSentences[i]))
        articleSentences[i] =  sentence
    
    while '' in articleSentences:
        articleSentences.remove('')

    
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


def clean(snippet, stopWords):
    import spacy
    nlp = spacy.load('en')
    # remove NE
    cleanedSnippets = []
    for s in snippet:
        neToRemove = set()
        #print (s)
        # stToRemove = set()
        doc = nlp(s)
        for ent in doc.ents:
            #print (ent.text)
            neToRemove.add(ent.text)
        neWordToRemove = set()
        for word in neToRemove:
            # cannot clean when left right space are other cases.
            # s = s.replace(' '+word+' ', ' ')
            if (len(word.split())>1):
                s = s.replace(word, '')
                s = s.replace(word+'\'s', '')
            else:
                neWordToRemove.add(word)
        '''
        for word in neWordToRemove:
            print (word)
        '''
        #print(s)
        sList = s.split(' ')
        cleanedSList = []
        # needs debug
        for word in sList:
            # deal with Obama's
            if word.lower() not in stopWords and word.replace('\'s', '') not in neWordToRemove:
                cleanedSList.append(word)
        cleanedSnippet = ' '.join(cleanedSList)
        # print (cleanedSnippet)
        cleanedSnippets.append(cleanedSnippet)
    return cleanedSnippets


if os.path.isfile(relatedSnippetsPath):
    return

claimFile = open(claimPath)
claims = []
with open(claimFile) as f:
    claims = f.readlines()
claims = [x.strip() for x in claims] 

articleFile = open(articlePath)
articles = []
with open(articleFile) as f:
    articles = f.readlines()
articles = [x.strip() for x in articles] 

articleLabels = np.load(credPath)


relatedSnippets = []
relatedSnippetLabels = []
#relatedSnippetMarkNumbers = []  # record number of ! ? "", list of list

from sklearn.feature_extraction.text import CountVectorizer
# empty string can be taken as all 0 vectors
# using both uni- and bi-grams
vectorizer = CountVectorizer(analyzer = "word", \
                            preprocessor = None, \
                            # should not use stop words, otherwise numbers and some other entities gets down.
                            # stop_words = 'english', \
                            ngram_range=(1, 2))
                             #max_features = 5000) 

from sklearn.metrics.pairwise import cosine_similarity

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

    '''
    # use lda to calculate similarity
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
    # print (similarityScore)
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
    relatedSnippet = [''.join(snippet) for snippet in snippets[overlapIdx].tolist()]
    relatedSnippets.extend(relatedSnippet)
    # relatedSnippetMarkNumbers.extend([snippetMarkNumbers[i] for i in overlapIdx])
    #print(relatedSnippets)
    #print(relatedSnippetLabels)

'''
stopWords read from file
handled by NER
'january','february','march','april','june','july','august','september','october','november','defnamecember',

too harsh
"every", "never", "whenever", "wherever", "whatever", "whoever", "anyhow", "anyway", "anywhere", "any", "always"

neg
'no', 'not'
'''
stopWords = []
with open('../../Data/stopword.txt') as f:
    stopWords = f.readlines()
    stopWords = [x.strip() for x in stopWords] 

relatedSnippets = clean(relatedSnippets, stopWords)

if not os.path.isfile(relatedSnippetsPath):
    with open(relatedSnippetsPath, 'w') as f:
        for item in relatedSnippets:
            f.write(item + '\n')
if not os.path.isfile(relatedSnippetLabelsPath + '.npy'):
    np.save(relatedSnippetLabelsPath, np.array(relatedSnippetLabels))


















