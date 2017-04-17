import scrapy
import json

class ArticlesSpider(scrapy.Spider):
    name = "articles_ver2"

    def __init__(self, aggregate_results_file=None, claim_file=None):
        with open(aggregate_results_file, 'r') as file:
            aggregate_results = json.load(file)
        claim_entry = next((c for c in aggregate_results if c['claim_file'] == claim_file), None)
        if claim_entry:
            self.start_urls = claim_entry['urls']
            self.claim = claim_entry['claim']
        else:
            self.start_urls = []


    def parse(self, response):

        yield {
            'claim':self.claim,
            'url':response.url,
            'html':response.body
        }