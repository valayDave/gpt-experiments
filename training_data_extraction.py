from source_analytics import *
from data_formatting import *

def iter_one(MAX_WORD_COUNT=1000,MIN_DOC_THRESHOLD=300,MIN_WORD_COUNT=500):
    sa = SourceAnalytics()
    df = sa.get_source_analytics()
    # Data should be scraped.
    df = df[df['scraped']==True]
    # Data Should be of certain world size larger than some size. 
    
    # Keep Data of Sources which are greater than 1000
    # Data Cleanup measure to remove text block under and image for associated press. 
    df['scraped_content'] = df['scraped_content'].str.replace(r"([^.]*?FILE - [^.]*\.)","")
    # Add formater which tokenizes things in the data for training. 
    formatter = SourceTextStyleFormater()
    # Prepare training data by applying formatter and cleaning the rest. 
    df['training_content'] = df.apply(lambda row: formatter(row['scraped_content'],row['title']),axis=1)
    df = df[df['training_content'].str.split().str.len() <= MAX_WORD_COUNT]
    # Filtering post cleanup.
    df = df[df['training_content'].str.split().str.len() >= MIN_WORD_COUNT]
    df = df.groupby('source.name').filter(lambda x: len(x) >= MIN_DOC_THRESHOLD)
    return df