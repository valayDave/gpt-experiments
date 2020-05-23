from source_analytics import *

def iter_one(MIN_CHAR_LENGTH=2000,MIN_DOC_THRESHOLD=900):
    sa = SourceAnalytics()
    df = sa.get_source_analytics()
    # Data should be scraped.
    df = df[df['scraped']==True]
    # Data larger than some size. 
    df = df[df['scraped_content'].str.len() > MIN_CHAR_LENGTH]
    # Keep Data of Sources which are greater than 1000
    df = df.groupby('source.name').filter(lambda x: len(x) >= MIN_DOC_THRESHOLD)
    return df