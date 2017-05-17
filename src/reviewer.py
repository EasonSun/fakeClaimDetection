from google import google
from urllib2 import urlopen
from bs4 import BeautifulSoup as bs
from bs4.element import Comment
from urlparse import urlparse
import time
import pickle

from Classifier import Classifier
from relatedSnippetsExtractor import relatedSnippetsExtractor

badTags = set(['a', 'img', 'style', 'script', '[document]', 'head', 'title', 'link', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'time', 'tr'])

class Reviewer(object):
    """docstring for Reviewer"""
    def __init__(self, query, sourcePath, doc2vecPath, logPath, experimentPath):
        self.query = query
        self.sources = []
        self.articles = []
        self.sourceMatrix = pickle.load(io.open(sourcePath, 'rb'))
        self.rsExtractor = relatedSnippetsExtractor(overlapThreshold, doc2vecPath=doc2vecPath)
        self.stanceClf = Classifier('stance', logPath, experimentPath)

        self.search()

    def search(self):
        num_page = 3
        search_results = google.search(self.query, num_page)
        articles = []
        sources = []
        for result in search_results:
            self.sources.append('{uri.netloc}'.format(uri=urlparse(result.link)))
            try:
                soup = bs(urlopen(result.link).read(), 'lxml')
            except:
                continue
            texts = soup.findAll(text=True)
            self.articles.append(filter(self.visible, texts))

    def visible(self, element):
        if element.parent.name in badTags:
            return False
        elif isinstance(element, Comment):
            return False
        return True
    
    def review(self):
        for article, source in zip(articles, sources):
        _, relatedSnippetsX_, relatedSnippets_, _, overlapScores_ = rsExtractor.extract(claim, article)
        # can be many other edge cases
        if relatedSnippets_ is not None:
            stanceProb_ = stanceClf.predict_porb(relatedSnippetsX_)
            del relatedSnippetsX_
            stanceScore_ = stanceProb_ * overlapScores_
            posTK10.add(stanceScore_[:,0])
            negTK10.add(stanceScore_[:,1])
            articlesScore.append((posTK10.avg(), negTK10.avg()))
            updateSource ((posTK10.avg(), negTK10.avg()), source, cred)
            relatedSnippets.extend(relatedSnippets_)
            del relatedSnippets_
            posStanceScores.extend(list(stanceScore_[:,0]))
            negStanceScores.extend(list(stanceScore_[:,1])) 
        else:
            # alert the user about that no related snippets are returned.
            return

# this part also needs parellel
t1 = time.time()
reviewer = Reviewer("DNC staffer Seth Rich sent 'thousands of leaked e-mails' to WikiLeaks before he was murdered.")
print (time.time() - t1)

reviewer.review()