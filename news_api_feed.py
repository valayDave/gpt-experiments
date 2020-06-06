

from newsapi import NewsApiClient
import os
import json
DATA_ROOT_DIR = './data'
ARTICLE_LIST_FILE_NAME = 'articles.json'
CORE_SOURCES = [
    'abc-news',
    'al-jazeera-english',
    'associated-press',
    'bbc-news',
    'bloomberg',
    'breitbart-news',
    'buzzfeed',
    'cbc-news', 
    'cbs-news',
    'cnn',
    'fox-news',
    'independent',
    'msnbc',
    'nbc-news',
    'politico',  # Completed Till Here. 
    'reuters',
    'the-huffington-post',
    'the-wall-street-journal',
    'the-washington-post',
    'the-washington-times',
    'time',
    'usa-today',
    'vice-news',
    'wired'
]


CURR_SOURCES = [
    'abc-news',
    'associated-press',
    'breitbart-news',
    'bbc-news',
    # 'bloomberg',
    # 'al-jazeera-english',
    # 'buzzfeed',
    # 'cbc-news', 
    # 'cbs-news',
    # 'cnn',
    # 'fox-news',
    # 'independent',
    # 'msnbc',
    # 'nbc-news',
    # 'politico',
]

QUERYING_DATES = [    
    '2020-04-22',
    '2020-04-23',
    '2020-04-24',
    '2020-04-25',
    '2020-04-26',
    '2020-04-27',
    '2020-04-28',
    '2020-04-29',
    '2020-04-30',
    '2020-05-01',
    '2020-05-02',
    '2020-05-03',
    '2020-05-04',
    '2020-05-05',
    '2020-05-06',
    '2020-05-07',
    '2020-05-08',
    '2020-05-09',
    '2020-05-10',
    '2020-05-11',
    '2020-05-12',
    '2020-05-13',
    '2020-05-14',
    '2020-05-15',
    '2020-05-16',
    '2020-05-17',
    '2020-05-18',
    '2020-05-19',
    '2020-05-20',
    '2020-05-21',
    '2020-05-22',
    '2020-05-23', # Start scraping from here. 
    '2020-05-24',
    '2020-05-25',
    '2020-05-26',
    '2020-05-27',
    '2020-05-28',
    '2020-05-29',
    '2020-05-30',
    '2020-05-31',
    '2020-06-01',
    
]

class SourceExtractor():
    """SourceExtractor 
    Extracts the data from the NewsFeed API. 
    """
    def __init__(self,source_name,api_key=None,data_root_dir=DATA_ROOT_DIR):
        self.name = source_name
        self.api_key = api_key   
        self.data_root_dir = data_root_dir
        self.articles = []
        
    def articles_present(self,from_date):
        file_path = os.path.join(self._get_dir_path(from_date),ARTICLE_LIST_FILE_NAME)
        try:
            os.stat(file_path)
            return True
        except:
            return False

    def get_articles(self,from_date,cache=True,ignore_none=False):
        """get_articles 
        Extracts articles according to sources. From File System 
        if cache=True:
            self.articles = articles
        """
        file_path = os.path.join(self._get_dir_path(from_date),ARTICLE_LIST_FILE_NAME)
        if not self.articles_present(from_date):
            if not ignore_none:
                raise Exception("No Articles Present For %s" % file_path)
            else:
                return []


        with open(file_path,'r') as json_file:
            articles = json.load(json_file)
        if cache:
            self.articles += articles

        return articles

    def query_source_remote(self,from_date,save_to_file=False):
        """query_source_remote 
        Extracts articles on `from_date`. It Uses NewApI to extract the data.
        This is Ment to be use for Querying and Storage 
        if save_to_file=True
            Saves files to DATA_ROOT_DIR/DATE/SOURCE_NAME/articles.json
        :type from_date: [2020-05-21]
        :param save_to_file: : Will Save to FS otherwise no
        :raises E: When there is API Limits reached etc. 
        :return: List[Articles] : see NewApi Docs for structure of Article Json. 
        """
        try:
            articles = self._query_remote(from_date=from_date,to_date=from_date)
        except Exception as E:
            raise E # Raise exception because there is no 

        if save_to_file:
            for article in articles: article['scraped'] = False
            self._save_to_file(from_date,articles)
        
        return articles        

    def _query_remote(self,from_date='',to_date=''):
        if not self.api_key:
            raise Exception("No API Key to NewsAPI")
        newsapi = NewsApiClient(api_key=self.api_key)
        articles = []
        first_call_response = newsapi.get_everything(from_param=from_date,to=to_date,sort_by='publishedAt',sources=self.name,page_size=100)
        # total_documents = first_call_response['totalResults']
        articles += first_call_response['articles']
        return articles

    def _get_dir_path(self,from_date):
        return os.path.join(self.data_root_dir,from_date,self.name)

    def _save_to_file(self,from_date,articles):
        """_save_to_file 
        Saves articles according to a date to directory structure DATA_ROOT_DIR/DATE/SOURCE_NAME/articles.json
        """
        # $ Create a folder path : ./data/date/publisher_name
        file_save_path = self._get_dir_path(from_date)
        try:
            os.makedirs(file_save_path)
        except:
            pass
        # $ Change path to file path from folder path
        file_save_path = os.path.join(file_save_path,ARTICLE_LIST_FILE_NAME)
        with open(file_save_path,'w') as json_file:
            json.dump(articles,json_file)
        