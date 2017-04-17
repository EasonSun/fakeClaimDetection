# Usage
To extract html bodies that correspond to claims in `[<lower_group_id>_aggregate_results.json, ..., <higher_group_id>_aggregate_results.json]` from `aggregate_results_groups` folder

`python run.py <lower_group_id> <higher_group_id>`

Examples: 

`python run.py 1` 

`python run.py 1 10` (Extract raw html for 1_aggregate_results.json, 2_..., 3_..., ..., 10_aggregate_results.json)

# Files
`article_spider_ver2.py` the spidert that extracts html body

`run.py` it is used to initiate `scrapy runspider` command

`split_into_groups.py` it splits `aggregate_results.json` into roughly 300 groups in `aggregate_results_groups` folder

# Results
## Extracted raw HTML
`final_results_ver2` folder