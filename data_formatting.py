import re
class TextStyleFormater():
    def __init__(self):
        self.paragraph_token = "PARAGRAH"
        self.newline_token = "NEW_LINE"
    
    def _quote_token(self,token):
        return " <"+token+"> "
    
    def _unquote_token(self,token):
        return " <"+token+"/> "
    
    def format(self,content_text,content_headline):
        raise NotImplementedError
    
    def get_all_tokens(self):
        tokens = [
            self._quote_token(self.paragraph_token),\
            self._quote_token(self.newline_token)\
        ]
        return tokens

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
    
    @staticmethod
    def clean_ascii(text):
        # function to remove non-ASCII chars from data
        return ''.join(i for i in text if ord(i) < 128)

    def get_all_tokens(self):
        tokens = super().get_all_tokens()
        curr_tokens = [
            self._quote_token(self.headline_token),\
            self._unquote_token(self.headline_token),\
            self._quote_token(self.body_token),\
            self._unquote_token(self.body_token),\
        ]
        return_tokens = tokens + curr_tokens
        return_tokens = [token.strip() for token in return_tokens]
        return return_tokens

    def format(self, content_text, content_headline,remove_paragraphs=None):
        content_text = content_text.replace('\n\n',self._quote_token(self.paragraph_token))
        if remove_paragraphs is not None:
            shortend_text = content_text.split(self._quote_token(self.paragraph_token))[:-remove_paragraphs]
            content_text = self._quote_token(self.paragraph_token).join(shortend_text)
            
        content_text = content_text.replace('\n',self._quote_token(self.newline_token))

        final_text = [
            self._quote_token(self.headline_token),\
            '' if not content_headline else content_headline,\
            self._unquote_token(self.headline_token),\
            self._quote_token(self.newline_token),\
            self._quote_token(self.body_token),\
            self.clean_ascii(content_text),\
            self._unquote_token(self.body_token),\
        ]
        return ''.join(final_text)
    
    def decompile(self,formatted_text):
        formatted_text = formatted_text.replace(self._quote_token(self.paragraph_token),'\n\n')
        formatted_text = formatted_text.replace(self._quote_token(self.newline_token),'\n')
        headline_text = re.findall(r'<HEADLINE>(.*)<HEADLINE/>',formatted_text)[0]
        formatted_text = re.sub(r'<HEADLINE>(.*)<HEADLINE/>','',formatted_text) 
        
        formatted_text = formatted_text.replace(self._quote_token(self.body_token),'')
        formatted_text= formatted_text.replace(self._unquote_token(self.body_token),'')
        
        return (headline_text,formatted_text)
