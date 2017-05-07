## Usage
### Set up enviornment
`pip install virtualenv virtualenvwrapper`

open `~/.bash_profile`

add this two lines

`# Virtualenv/VirtualenvWrapper`
`source /usr/local/bin/virtualenvwrapper.sh`

find your Python2 path: `which python`

Start virtualenv

`mkvirtualenv -p [your Python2 path] article_extraction/`

Then 

`pip install -r requirements.txt`

You are good to run the scripts. 

To shut down the virtualenv:

`deactivate`

To go back to this virtualenv:

`workon article_extraction`


### Run the script
Extract html bodies (raw articles) about each claim in groups:

`python run.py <lower_group_id> [<higher_group_id>]`

Examples: 

Extract raw html from group 1

`python run.py 1` 

Extract raw html from group 1 to group 10

`python run.py 1 10`

The claims and links of the articles about this claim are seperated in the `aggregate_results_groups` folder


## Files
`article_spider_ver2.py` the spidert that extracts html body

`run.py` it is used to initiate `scrapy runspider` command

`split_into_groups.py` it splits `aggregate_results.json` into roughly 300 groups in `aggregate_results_groups` folder

## Results
### Extracted raw HTML
`final_results_ver2` folder

### Result Data Format
```
{
"Claim": "blahblah",  # the same field as in the Snopes
"Credibility": "true",  # the same field as in the Snopes 
"Article": [[], [], [], ...],  # a list of list, each inner list is 30 articles for a claim (excluding Snopes domain)
"Source": [[], [], [], ...],   # a list of list, each inner list is the 30 url of articles for a claim (1-1 mapping with Article)
}
```