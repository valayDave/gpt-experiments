
import time
from news_api_feed import *
from article_scraper import ArticleScraper
import multiprocessing
import click 

NUM_SUBPROCESSES = multiprocessing.cpu_count()
CURR_SOURCES = [
    'abc-news'
    'al-jazeera-english',
    'associated-press',
    'bbc-news',
    'bloomberg',
    'breitbart-news',
    'buzzfeed'
    ]

@click.group()
def cli():
  pass


def scrape_articles(date,source_extractor:SourceExtractor):
    scraper = ArticleScraper(date,source_extractor)    
    num_scraped = scraper.run()
    return num_scraped

def print_result(result):
    print(result)    

@cli.command(name='scrape_articles')
@click.argument('num_processes', type=int,default=4)
@click.option('--max_ext',default=None, type=int, help="Maximum Total Scraping Process Opens")
def multiprocess_scraping(num_processes,max_ext,sources=CURR_SOURCES):
    pool = multiprocessing.Pool(num_processes)
    num_procs = 0
    
    def check_max(max_ext,num_procs):
        if not max_ext:
            return False
        return max_ext < num_procs

    for date in QUERYING_DATES:
        if check_max(max_ext, num_procs):
            break
        for source in sources:
            if check_max(max_ext, num_procs):
                break
            source_ext = SourceExtractor(source)
            if not source_ext.articles_present(date):
                continue
            pool.apply_async(scrape_articles,(date,source_ext),callback=print_result)
            num_procs+=1
    
    pool.close()
    pool.join()
     
@cli.command(name='scrape_sources')
@click.argument('news_api_key', type=str,default='')
def extract_source_data(news_api_key):
    print(news_api_key)
    for source in CORE_SOURCES:
        print(source)
        ext = SourceExtractor(source,api_key=news_api_key)
        for query_date in QUERYING_DATES:
            time.sleep(0.1)
            try: 
                ext.query_source_remote(query_date,save_to_file=True)
            except Exception as e:
                print(e)
                break

if __name__=='__main__':
    cli()