import scrapy
from items import ArticleExtractionItem
from scrapy.crawler import CrawlerProcess
import json
from scrapy.utils.project import get_project_settings
from twisted.internet import reactor, defer
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging

class ArticlesSpider(scrapy.Spider):
    name = "articles"

    custom_settings = {
        'ITEM_PIPELINE': {
            'article_extraction.pipelines.ArticleExtractionPipeline':100
        }
    }

    def __init__(self, claim=None, claim_filename=None, urls=None, *args, **kwargs):
        super(ArticlesSpider, self).__init__(*args, **kwargs)
        self.claim = claim
        self.claim_filename = claim_filename
        self.urls = urls

        self.start_urls = self.urls

    def parse(self, response):
        self.log(self.claim)
        article = response.xpath('//text()').extract()
        article = ' '.join(article)
        item = ArticleExtractionItem()
        item['claim'] = self.claim
        item['claim_filename'] = self.claim_filename
        item['article'] = article
        return item


if __name__ == "__main__":
    claim_and_urls_file = 'aggregate_results.json'
    # # process = CrawlerProcess(get_project_settings())
    # with open(claim_and_urls_file, 'r') as file:
    #     claims = json.load(file)
    #     for claim in claims:
    #         #spider = ArticlesSpider(claim=claim['claim'], claim_filename=claim['claim_file'], urls=claim['urls'])
    #         # ArticlesSpider.claim = claim['claim']
    #         # ArticlesSpider.claim_filename = claim['claim_file']
    #         # ArticlesSpider.start_urls = claim['urls']
    #         process.crawl(ArticlesSpider, claim=claim['claim'], claim_filename=claim['claim_file'], urls=claim['urls'] )

    # process.start()

    configure_logging()
    runner = CrawlerRunner(get_project_settings())


    @defer.inlineCallbacks
    def crawl():
        claims = []
        with open(claim_and_urls_file, 'r') as file:
            claims = json.load(file)
        for claim in claims:
            yield runner.crawl(ArticlesSpider, claim=claim['claim'], claim_filename=claim['claim_file'],
                              urls=claim['urls'])

        reactor.stop()


    crawl()
    reactor.run()

