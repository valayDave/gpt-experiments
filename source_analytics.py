from news_api_feed import *
import pandas
import re

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
                self.source_analytics.append(SourceDateAnalytics(ext.get_articles(date,cache=False,ignore_none=True),source,date))
    
    def get_source_analytics(self):
        sources = []
        for src in self.source_analytics:
            sources+=src.articles
        df = pandas.DataFrame(pandas.json_normalize(sources))
        return df
    
    def __str__(self):
        return '''

        '''

class TextStyleFormater():
    def __init__(self):
        self.paragraph_token = "PARAGRAH"
        self.newline_token = "NEW_LINE"
    
    def _quote_token(self,token):
        return "<"+token+">"
    
    def _unquote_token(self,token):
        return "<"+token+"/>"
    
    def format(self,content_text,content_headline):
        raise NotImplementedError

    def __call__(self,content_text,content_headline):
        return self.format(content_text,content_headline)

    def decompile(self,formatted_text):
        raise NotImplementedError

class SourceTextStyleFormater(TextStyleFormater):
    def __init__(self):
        TextStyleFormater.__init__(self)
        self.headline_token = "HEADLINE"
        self.body_token = "BODY"
        # self.source = source # integrate source later
        # self.source_token = "<|"+str(source_name).capitalize()+"|>" # dont need this right now. 
    
    def format(self, content_text, content_headline):
        content_text = content_text.replace('\n\n',self._quote_token(self.paragraph_token))
        content_text = content_text.replace('\n',self._quote_token(self.newline_token))

        final_text = [
            self._quote_token(self.headline_token),\
            '' if not content_headline else content_headline,\
            self._unquote_token(self.headline_token),\
            self._quote_token(self.newline_token),\
            self._quote_token(self.body_token),\
            content_text,\
            self._unquote_token(self.body_token),\
        ]
        print(len(final_text))
        return ''.join(final_text)
    
    def decompile(self,formatted_text):
        formatted_text = formatted_text.replace(self._quote_token(self.paragraph_token),'\n\n')
        formatted_text = formatted_text.replace(self._quote_token(self.newline_token),'\n')
        headline_text = re.findall(r'<HEADLINE>(.*)<HEADLINE/>',formatted_text)[0]
        formatted_text = re.sub(r'<HEADLINE>(.*)<HEADLINE/>','',formatted_text) 
        
        formatted_text = formatted_text.replace(self._quote_token(self.body_token),'')
        formatted_text= formatted_text.replace(self._unquote_token(self.body_token),'')
        
        return (headline_text,formatted_text)



class StyleTransferPreprocessor():
    
    def __init__(self):
        # This will ensure that all signatures regarding cleaning up picture related-text/ 
        self.photo_cleaner = True


    def _clear_photo_meta(self,text_content):
        # Remove the picture related content to see how it will behave 
        # associated-press pattern for photo credits
        ap_pattern_text = re.findall(r"([^.]*?FILE-[^.]*\.)",text_content)
        
        # todo : remove the pattern detected in the text. 
    
