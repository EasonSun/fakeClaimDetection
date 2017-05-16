import numpy as np
import re
import sys
import os 
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import time
import io
from gensim import models


#from overlap_lsi import overlapping


'''
stopWords read from file, because the sklearn one is not enough
handled by NER
'january','february','march','april','june','july','august','september','october','november','defnamecember',

too harsh
"every", "never", "whenever", "wherever", "whatever", "whoever", "anyhow", "anyway", "anywhere", "any", "always"

neg
'no', 'not'
'''
contractions = re.compile(r"'|-|\"")
# all non alphanumeric
symbols = re.compile(r'(\W+)', re.U)
# single character removal
singles = re.compile(r'(\s\S\s)', re.I|re.U)
# separators (any whitespace)
seps = re.compile(r'\s+')

alteos = re.compile(r'([!\?])')

class relatedSnippetsExtractor(object):
    """docstring for ClassName"""
    def __init__(self, overlapThreshold, glovePath=None, doc2vecPath=None):
        self.overlapThreshold = overlapThreshold
        self.stopWords = []  
        try:
            f = io.open("data/stopword.txt")
        except FileNotFoundError:
            f = io.open("../data/stopword.txt")
        self.stopWords = f.readlines()
        self.stopWords = [x.strip() for x in self.stopWords] 
        f.close()
        if glovePath is not None:
            self.glove = pickle.load(io.open(glovePath, 'rb'))
            self.doc2vec = None
        if doc2vecPath is not None:
            self.doc2vec = models.Doc2Vec.load(doc2vecPath)
            self.glove = None
        print ("overlapThreshold = %f" %self.overlapThreshold)


    def extract(self, claim, article, label=None):
        #from sklearn.feature_extraction.text import CountVectorizer
        # empty string can be taken as all 0 vectors
        # using both uni- and bi-grams
        '''
        vectorizer = CountVectorizer(analyzer = "word", \
                                    preprocessor = None, \
                                    # watch out stop words, should not extract named entities!
                                    # possible number entities like sixty
                                    stop_words = 'english', \
                                    ngram_range=(1, 2))
                                     #max_features = 5000) 
        '''
        # print (article)
        # print (claim)
        claim = self._cleanText(claim)
        claimX = self._embed(claim.split())
        claimX = claimX.reshape(1, claimX.size)
        snippets, snippetsX = self._extractSnippets(article)
        if (snippets == [] or snippetsX is None):
            return None, None, None, None 
        similarityScore = cosine_similarity(claimX, snippetsX)[0]
        #del claimX
        #del snippetsX
        #print (similarityScore)
        if (np.count_nonzero(similarityScore) == 0):
            # bad and weird thing happens 
            return None, None, None, None
        minSimilarityScore = np.max(similarityScore[np.nonzero(similarityScore)])
        if (minSimilarityScore < self.overlapThreshold):
            return None, None, None, None
        # print (minSimilarityScore)
        overlapIdx = np.where(similarityScore > self.overlapThreshold)[0]
        #print (overlapIdx)
        #snippets = np.array([[snippet] for snippet in snippets])
        #print (snippets.shape)
        # from vector back to sentence to later use them in the same feature space
        #print (snippets)
        relatedSnippets = [' '.join(snippet) for snippet in np.array(snippets)[overlapIdx].tolist()]
        relatedSnippetsX = snippetsX[overlapIdx]
        del snippets
        # relatedSnippets = self._clean(relatedSnippets)
        relatedSnippetLabels = None
        if label is not None:
            relatedSnippetLabels = [label for i in range(len(overlapIdx))]
            # return a list of related snippets (str)
            # corresponding to a claim and an article
        return claimX, relatedSnippetsX , relatedSnippets, relatedSnippetLabels
        #print(relatedSnippets)
        #print(relatedSnippetLabels)


    def _extractSnippets(self, article):
        # a list of word list from a snippet
        snippets = []
        snippetsX = None
        # snippetMarkNumbers = []   #list of list, inner list records number of ! ? ""
        # number of a snippet in a sentence
        # should be the number to make stance classification best
        # but best distribution happens at 3 for google crawled
        NSS = 3 

        articleSentences = alteos.sub(r' \1 .', article).rstrip("(\.)*\n").split('.')
        ctr = 0
        snippet = ''
        for sen in articleSentences:
            if (len(sen.split())) > NSS:
                sen = self._cleanText(sen)
                if (len(sen.split())) > NSS:
                    if ctr < NSS:
                        snippet += (' ' + sen)
                        ctr += 1

            if ctr == NSS:
                snippets.append(snippet.split())
                if snippetsX is None:
                    snippetsX = self._embed(snippet.split())
                    snippetsX = snippetsX.reshape(1, snippetsX.size)
                else:
                    snippetsX = np.vstack((snippetsX, self._embed(snippet.split()))) 
                ctr = 0
                del snippet
                snippet = ''
        if ctr != 0:
            snippets.append(snippet.split())
            if snippetsX is None:
                    snippetsX = self._embed(snippet.split())
                    snippetsX = snippetsX.reshape(1, snippetsX.size)
            else:
                snippetsX = np.vstack((snippetsX, self._embed(snippet.split()))) 
        del snippet
        return snippets, snippetsX

        # articleSentences = re.split(r'[.|!|?]', article)
        
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

    def _cleanText(self, text):        
        # cleaner (order matters)
        text = text.lower()
        text = contractions.sub('', text)
        text = symbols.sub(r' \1 ', text)
        text = singles.sub(' ', text)
        text = seps.sub(' ', text)
        return text

    # take in a list of words
    def _embed(self, sentence):
        if self.glove is not None:
            vec = np.zeros((1,200))
            ctr = 0
            for word in sentence:
                if word in self.glove:
                    vec += self.glove[word]
                    ctr += 1
            if ctr != 0:
                return vec / ctr
            else:
                return vec

        elif self.doc2vec is not None:
            # defined by that paper
            start_alpha=0.01
            infer_epoch=1000
            # shape: (300,)
            return self.doc2vec.infer_vector(sentence, alpha=start_alpha, steps=infer_epoch)

        else:
            #BoW goes here!
            pass










