from google import google
from urllib2 import urlopen
from bs4 import BeautifulSoup as bs
from bs4.element import Comment
from urlparse import urlparse
import time

badTags = set(['a', 'img', 'style', 'script', '[document]', 'head', 'title', 'link', 'ul', 'ol', 'li', 'dl', 'dt', 'dd', 'time', 'tr'])

class Reviewer(object):
    """docstring for Reviewer"""
    def __init__(self, query):
        self.query = query
        self.sources = []
        self.articles = []
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
        pass
        #for article, source in zip(articles, sources):

# this part also needs parellel
t1 = time.time()
reviewer = Reviewer("DNC staffer Seth Rich sent 'thousands of leaked e-mails' to WikiLeaks before he was murdered.")
print (time.time() - t1)

reviewer.review()