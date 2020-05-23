from news_api_feed import *
import pandas

CURR_SOURCES = [
    'abc-news',
    'al-jazeera-english',
    'associated-press',
    'bbc-news',
    'bloomberg',
    'breitbart-news',
    'buzzfeed'
]

class SourceDateAnalytics():
    def __init__(self,articles,source,date):
        self.articles = articles
        self.date = date
        self.source = source
        
        # Analytics Properties
        self.scraped = self._get_scraped_count()
        self.error = self._get_error_count()

    def __len__(self):
        return len(self.articles)    
    
    def _get_error_count(self):
        err_count = 0
        for art_obj in self.articles:
            if 'error' not in art_obj:
                continue
            if art_obj['error']:
                err_count+=1
        return err_count

    def _get_scraped_count(self):
        scraped = sum([1 for art_obj in self.articles if art_obj['scraped']])
        return scraped


class SourceAnalytics():
    """SourceAnalytics
    Holds the Analytics of all `sources` according for all `dates`
    """
    def __init__(self,dates=QUERYING_DATES,sources=CURR_SOURCES):
        self.sources = sources
        self.dates = dates
        self.source_analytics = []
        self._load_all_data()
    
    def __len__(self):
        return sum([a.scraped for a in self.source_analytics])

    def _load_all_data(self):
        for source in self.sources:
            for date in self.dates:
                ext = SourceExtractor(source)
                self.source_analytics.append(SourceDateAnalytics(ext.get_articles(date,cache=False),source,date))
    
    def get_source_analytics(self):
        sources = []
        for src in self.source_analytics:
            sources+=src.articles
        df = pandas.DataFrame(pandas.json_normalize(sources))
        return df
    
    def __str__(self):
        return '''

        '''