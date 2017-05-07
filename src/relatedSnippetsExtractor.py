import numpy as np
import re
import sys
import os 

#from overlap_lsi import overlap
stopwordsPath="data/stopword.txt"
'''
stopWords read from file, because the sklearn one is not enough
handled by NER
'january','february','march','april','june','july','august','september','october','november','defnamecember',

too harsh
"every", "never", "whenever", "wherever", "whatever", "whoever", "anyhow", "anyway", "anywhere", "any", "always"

neg
'no', 'not'
'''


class relatedSnippetsExtractor(object):
    """docstring for ClassName"""
    def __init__(self, overlapThreshold):
        self.overlapThreshold = overlapThreshold
        self.stopWords = []  
        with open(stopwordsPath) as f:
            self.stopWords = f.readlines()
            self.stopWords = [x.strip() for x in self.stopWords] 
        print ("overlapThreshold = %f" %self.overlapThreshold)


    def extract(self, claim, article, label=None):
        from sklearn.feature_extraction.text import CountVectorizer
        # empty string can be taken as all 0 vectors
        # using both uni- and bi-grams
        vectorizer = CountVectorizer(analyzer = "word", \
                                    preprocessor = None, \
                                    # watch out stop words, should not extract named entities!
                                    # possible number entities like sixty
                                    stop_words = 'english', \
                                    ngram_range=(1, 2))
                                     #max_features = 5000) 

        from sklearn.metrics.pairwise import cosine_similarity
        # print (article)
        # print (claim)
        snippets = self._extractSnippets(article)
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

        overlapIdx = np.where(sims > self.overlapThreshold)[0]
        #print (overlapIdx)
        relatedSnippetLabels.extend([label for i in range(len(overlapIdx))])
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
        vocab = self._extractVocab([claim], snippets, vectorizer)
        if len(vocab.keys()) == 0:
            # bad thing can happen
            return None, None 
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
            return None, None
        minSimilarityScore = np.min(similarityScore[np.nonzero(similarityScore)])
        if (minSimilarityScore < self.overlapThreshold):
            return None, None
        # print (minSimilarityScore)
        similarityScore = np.zeros((1, snippetsX.shape[0]))

        numSnippets = snippetsX.shape[0]
        for i in range(0, numSnippets ,500):
            j = min(i+500, numSnippets-1)
            similarityScore[0][i:j] = cosine_similarity(snippetsX[i:j], claimX)[0]
        overlapIdx = np.where(similarityScore > self.overlapThreshold)[0]
        #print (overlapIdx)
        snippets = np.array([[snippet] for snippet in snippets])
        #print (snippets.shape)
        # from vector back to sentence to later use them in the same feature space
        relatedSnippets = [''.join(snippet) for snippet in snippets[overlapIdx].tolist()]
        relatedSnippets = self._clean(relatedSnippets)
        relatedSnippetLabels = None
        if label is not None:
            relatedSnippetLabels = [label for i in range(len(overlapIdx))]
            # return a list of related snippets (str)
            # corresponding to a claim and an article
        return relatedSnippets, relatedSnippetLabels
        #print(relatedSnippets)
        #print(relatedSnippetLabels)


    def _extractSnippets(self, article):
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
        for i in range(0, len(articleSentences) - NSS + 1, NSS):
            temp = articleSentences[i : i+NSS]
            snippet = ""
            for j in range(NSS):
                if snippet == '':
                    snippet += temp[j]
                else:
                    snippet += ' ' + temp[j]
            snippets.append(snippet)

            # grab non overlapping snippets
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


    def _extractVocab(self, claims, snippets, vectorizer):
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


    def _clean(self, snippets):
        import spacy
        nlp = spacy.load('en')
        # remove NE
        cleanedSnippets = []
        for s in snippets:
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
                if word.lower() not in self.stopWords \
                   and word.replace('\'s', '') not in neWordToRemove:
                    cleanedSList.append(word)
            cleanedSnippet = ' '.join(cleanedSList)
            # print (cleanedSnippet)
            cleanedSnippets.append(cleanedSnippet)
        return cleanedSnippets

    # BoW features. Shit
    def extractFeatures(self, relatedSnippets, MIN_DF, MAX_DF):
            from sklearn.feature_extraction.text import TfidfVectorizer
            # empty string can be taken as all 0 vectors
            # using both uni- and bi-grams
            vectorizer = TfidfVectorizer(analyzer = "word", \
                                         stop_words = "english",   \
                                         tokenizer = None,    \
                                         preprocessor = None, \
                                         min_df=MIN_DF, \
                                         max_df=MAX_DF, \
                                         ngram_range=(1, 2))    
            '''
            the min df above is really important as the first step for fieature engineering
            .005 means only keep features apper more than .005 portion of docs
            that is roughly 486 docs
            '''
            relatedSnippetX = vectorizer.fit_transform(relatedSnippets)
            # print (vectorizer.vocabulary_)
            relatedSnippetX = relatedSnippetX.toarray()
            featureNames = vectorizer.get_feature_names()
            assert(relatedSnippetX.shape[1] == len(featureNames))
            return relatedSnippetX, featureNames

            #relatedSnippetMarkNumberX = np.array(relatedSnippetMarkNumbers)
            #np.save('relatedSnippetMarkNumberX', relatedSnippetMarkNumberX)

            # print("relatedSnippetX dim and relatedSnippet_y dim: ")
            # print(relatedSnippetX.shape, relatedSnippet_y.shape)















