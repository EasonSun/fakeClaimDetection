# Format
```
{
"Claim": "blahblah",  # the same field as in the Snopes
"Credibility": "true",  # the same field as in the Snopes 
"Article": [[], [], [], ...],  # a list of list, each inner list is 30 articles for a claim (excluding Snopes domain)
"Source": [[], [], [], ...],   # a list of list, each inner list is the 30 url of articles for a claim (1-1 mapping with Article)
}
```

# How to run
`cd article_extraction\spiders`

`python articles_spider.py`

# Caution

Make sure `article_extraction_spiders/final_results` exists before running

Make sure `aggregate_results.json` exists before running

# Results

Extracted articles are stored in `article_extraction_spiders/final_results`
