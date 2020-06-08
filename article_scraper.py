import multiprocessing
from news_api_feed import *
import newspaper

class ArticleScraper():
    """ArticleScraper 
    
    Extract articles For a date and Publisher and Saves the file Set by the SourceExtractor. 
    """
    def __init__(self,date,article_extractor:SourceExtractor):
        # multiprocessing.Process.__init__(self)
        self.date=date
        self.article_extractor = article_extractor
        self.articles = self.article_extractor.get_articles(date)

    def run(self):
        print("Starting Scraping Content For , ",self.article_extractor.name,self.date)
        num_scraped = 0
        for article_obj in self.articles: 
            if article_obj['scraped']:
                continue
            art = newspaper.Article(article_obj['url']) 
            try: 
                art.build() 
            except: 
                article_obj['error'] = True
                continue 
            article_obj['scraped_content'] = art.text 
            article_obj['scraped'] = True
            article_obj['error'] = False
            num_scraped+=1
        print("Scraped ",num_scraped," Articles for ",self.article_extractor.name,self.date)
        self.article_extractor._save_to_file(self.date,self.articles)
        return num_scraped


def get_scraped_article(url):
    art = newspaper.Article(url)
    try: 
        art.build() 
    except:
        return None
    
    return {
        'content_text' : art.text,
        'title' : art.title
    }