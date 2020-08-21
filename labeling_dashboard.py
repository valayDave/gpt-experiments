import streamlit as st
from news_api_feed import *
import pandas
from language_model_tools import SourcePredictionModel
import torch 
import plotly
import plotly.graph_objects as go
import article_scraper

# Download a single file and make its content available as a string.
# @st.cache(show_spinner=False)
# def get_file_content_as_string(path):
#     url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
#     response = urllib.request.urlopen(url)
#     return response.read().decode("utf-8")

import numpy as np
OPTIONS = [
    'SEE_DATA',
    'SCRAPING_ANALYSIS',
]


@st.cache(show_spinner=True)
def get_data(source_name,selected_date):
    ext = SourceExtractor(source_name)
    art_objs = ext.get_articles(selected_date)
    df = pandas.DataFrame(pandas.json_normalize(art_objs))
    df['publishedAt'] = pandas.to_datetime(df['publishedAt'])
    return df

@st.cache(show_spinner=True,hash_funcs={SourcePredictionModel: id})
def get_model():
    storage_path = 'storage/models/classifier/1591355104/8'
    model_path = storage_path
    SourceModel = SourcePredictionModel(model_path)
    return SourceModel

# CORE_SOURCE_MODEL = get_model()


def round_tensor(tensor,n_digits=5):
    rounded = torch.round(tensor * 10**n_digits) / (10**n_digits)
    return rounded


def source_lookup():
    """source_lookup 
    This will hold all the needed functions for making the view. 
    """
    source_name = st.selectbox("Search for which Source?", CURR_SOURCES,1)
    selected_date = st.selectbox("Search for which Date?", QUERYING_DATES,1)
    scraped_toggle = st.checkbox("Show Only Scraped Content")

    source_model = get_model()

    df = get_data(source_name,selected_date)
    if scraped_toggle:
        df = df[df['scraped']==True]
    hist_values = np.histogram(df['publishedAt'].dt.hour,bins=24,range=(0,24))[0]
    title_search_text = st.text_input('Search for Stories From Title', '')
    regex_toggle = st.checkbox("Treat Search as Regex")
    st.header('Histogram Of Stories Published On : %s By %s'%(selected_date,source_name))
    #
    st.bar_chart(hist_values)
    # st.table(df[['title','publishedAt','author','source.id','source.name','scraped','error','author']].head(n=20))
    if len(df) == 0:
        return 
    
    search_df = df
    if title_search_text != '':
        search_df = df[df['title'].str.contains(title_search_text,case=regex_toggle)]
    ids = list(range(len(search_df)))
    if len(ids) ==0 :
        st.markdown("### No Content Found")
        return 
    value = st.selectbox("Read Any Story From Selected Ones", ids, format_func=lambda x: search_df.iloc[x]['title'])
    remove_paragraph_value = st.number_input('Select Number of Paragraphs to Remove',value=0)
    para_remove_parser = st.checkbox("Render Parsed Document Post Paragraph Removal")
    hide_model_prediction = st.checkbox("Hide Language Model Prediction")
    content = search_df.iloc[value]['content']
    if 'scraped_content' in df.iloc[value]:
        content = search_df.iloc[value]['scraped_content']
    
    headline = search_df.iloc[value]['title']
    
    if remove_paragraph_value == 0:
        remove_paragraph = None
    else:
        remove_paragraph = remove_paragraph_value
    
    model_predicted_source,source_likehood = source_model.predict_top_named_source(headline,\
                                        content,\
                                        remove_paragraphs=remove_paragraph)
    language_model_predictions = """
    ### Language Model Data 
    Predicted Source : *{model_predicted_source}*\n
    Predicted Score : {source_likehood}\n
    """.format(model_predicted_source=model_predicted_source,\
                source_likehood=round(float(source_likehood),6))                                    
    # st.markdown(str(source_model.column_split_order))    
    markdown_content = '''
    # {title}
    '''.format(title=headline)
    st.markdown('%s'%markdown_content)
    if not hide_model_prediction:
        st.markdown(language_model_predictions)
    st.markdown("### Content Scraped : %s" % 'True' if df.iloc[value]['scraped'] else 'False')
    if para_remove_parser:
        parsed_content = source_model._get_formatted_text(headline,content,remove_paragraphs=remove_paragraph) 
        st.markdown("""
        ## Parsed Content \n
        {parsed_content}\n
        ## Actual Content\n
        """.format(parsed_content=parsed_content))
    st.markdown(content)    


def training_data_lookup():
    from training_data_extraction import iter_one as train_filtered_df_generator
    df = train_filtered_df_generator()
    df['publishedAt'] = pandas.to_datetime(df['publishedAt'])
    
    plotting_df = df.groupby([df['publishedAt'].dt.strftime('%Y-%m-%d'),'source.id']).agg(['count']).reset_index()
    plotting_df

    sources = list(df['source.id'].unique())
    source_name = st.selectbox("Search for which Source?", sources,1)
    selected_date = st.selectbox("Search for which Date?", QUERYING_DATES,1)
    df = df[df['source.id']==source_name]
    
    # Filter Date
    df = df[df['publishedAt'].dt.strftime('%Y-%m-%d') == selected_date]
    
    
    # st.area_chart(plotting_df)
    st.bar_chart(plotting_df['publishedAt','source.id'])
    
    ids = list(range(len(df)))
    value = st.selectbox("Read Any Story From Selected Ones", ids, format_func=lambda x: df.iloc[x]['title'])

    content = df.iloc[value]['content']
    if 'scraped_content' in df.iloc[value]:
        content = df.iloc[value]['scraped_content']

    markdown_content = '''
    # {title}
    '''.format(title=df.iloc[value]['title'])
    st.markdown('%s'%markdown_content)
    st.markdown("### Content Scraped : %s" % 'True' if df.iloc[value]['scraped'] else 'False')
    st.markdown(content)    


def instant_page_lookup():
    source_model = get_model()
    st.markdown('# GPT-2 Publisher Realtime Classification Demo\n')
    st.sidebar.markdown(
        '''
        The current GPT-2 Model has been trained to classify {publishers}\n

        Give Link to URL from website and See It in action\n

        Example URL :\n
        [Brietbart Article](https://www.breitbart.com/2020-election/2020/08/20/live-updates-democratic-national-convention-night-four/)\n
        [BBC-News](https://www.bbc.com/news/world-europe-53856609)\n
        [ABC-News](https://abcnews.go.com/Politics/mom-traveling-kids-kicked-off-flight-year-refuses/story?id=72503986)\n
        [Associated-Press](https://apnews.com/a1909f485327e9aff894f6258a25e7af)\n
        '''.format(
            publishers =', '.join(source_model.column_split_order)
        )
    )
    title_search_text = st.text_input('Scraped Content From Given Link and Run classifier', '')
    if title_search_text == "":
        return 
    
    returned_data = article_scraper.get_scraped_article(title_search_text)
    if returned_data is None:
        error_message = '''
        ## Error Extracting Data For Article \n
        {article}
        '''.format(article=title_search_text)
        st.markdown(error_message)
        return 
    headline = returned_data['title']
    content = returned_data['content_text']
    remove_paragraph_value = st.number_input('Select Number of Paragraphs to Remove',value=0)
    if remove_paragraph_value == 0:
        remove_paragraph = None
    else:
        remove_paragraph = remove_paragraph_value

    model_predicted_source,source_likehood = source_model.predict_top_named_source(headline,\
                                        content,\
                                        remove_paragraphs=remove_paragraph)
    all_pred_tuples = source_model.get_all_predictions(headline,content,remove_paragraphs=remove_paragraph)
    language_model_predictions = """
    Predicted Source : *{model_predicted_source}*\n
    Predicted Score : {source_likehood}\n
    """.format(model_predicted_source=model_predicted_source,\
                source_likehood=round(float(source_likehood),6))
    # st.markdown(str(source_model.column_split_order))    
    df2=pandas.DataFrame(list(map(lambda x:{'source_name':x[0],'precent_prediction':x[1]},all_pred_tuples)))
    
    pub_dist_chart = go.Figure()
    pub_dist_chart.add_trace(
                go.Bar(x=df2['source_name'],\
                        y=df2['precent_prediction'],\
                        name='Probability'
                        )
            )
   
    markdown_content = '''
    # {title}\n
    ## Language Model Predictions
    {language_model_predictions}\n
    
    ## Publish Confidence Distribution
    '''.format(title=headline,language_model_predictions=language_model_predictions)
    st.markdown('%s'%markdown_content)

    st.plotly_chart(
        pub_dist_chart
    )
    st.markdown('## Content')
    # if para_remove_parser:
    #     parsed_content = source_model._get_formatted_text(headline,content,remove_paragraphs=remove_paragraph) 
    #     st.markdown("""
    #     ## Parsed Content \n
    #     {parsed_content}\n
    #     ## Actual Content\n
    #     """.format(parsed_content=parsed_content))
    st.markdown(content)  

    pass

def init_app():
    # st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        [
            # "Date Based Lookup", 
            # "Overall Analytics",
            "Instant Page Lookup"
            # "Training Data Lookup"
        ])
    
    if app_mode == "Date Based Lookup":
        source_lookup()
    elif app_mode == "Overall Analytics":
        st.code(open('labeling_dashboard.py').read())
    elif app_mode == "Instant Page Lookup":
        instant_page_lookup()
    # elif app_mode=='Training Data Lookup':
    #     training_data_lookup()
    # # if app_mode == "Show instructions":
    # #     st.sidebar.success('To continue select "Run the app".')
    # # elif app_mode == "Show the source code":
    # #     readme_text.empty()
    # #     st.code(get_file_content_as_string("app.py"))
    # # elif app_mode == "Run the app":
    # #     readme_text.empty()
    # #     run_the_app()
    # # Add a selectbox to the sidebar:

    # my_slot1 = st.empty()
    # # Appends an empty slot to the app. We'll use this later.

    # my_slot2 = st.empty()
    # # Appends another empty slot.

    # add_selectbox = my_slot2.selectbox(
    #     'How would you like to be contacted?',
    #     ('Email', 'Home phone', 'Mobile phone')
    # )

    # # Add a slider to the sidebar:
    # add_slider = my_slot1.slider(
    #     'Select a range of values',
    #     0.0, 100.0, (25.0, 75.0)
    # )
    
if __name__=='__main__':
    init_app()