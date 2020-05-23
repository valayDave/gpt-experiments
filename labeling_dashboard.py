import streamlit as st
from news_api_feed import *
import pandas

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

@st.cache(show_spinner=False)
def get_data(source_name,selected_date):
    ext = SourceExtractor(source_name)
    art_objs = ext.get_articles(selected_date)
    df = pandas.DataFrame(pandas.json_normalize(art_objs))
    df['publishedAt'] = pandas.to_datetime(df['publishedAt'])
    return df



def source_lookup():
    """source_lookup 
    This will hold all the needed functions for making the view. 
    """
    source_name = st.selectbox("Search for which Source?", CURR_SOURCES,1)
    selected_date = st.selectbox("Search for which Date?", QUERYING_DATES,1)
    scraped_toggle = st.checkbox("Show Only Scraped Content")
    
    # 
    df = get_data(source_name,selected_date)
    if scraped_toggle:
        df = df[df['scraped']==True]
    hist_values = np.histogram(df['publishedAt'].dt.hour,bins=24,range=(0,24))[0]
    st.header('Histogram Of Stories Published On : %s By %s'%(selected_date,source_name))
    #
    st.bar_chart(hist_values)
    # st.table(df[['title','publishedAt','author','source.id','source.name','scraped','error','author']].head(n=20))
    if len(df) == 0:
        return 
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

def init_app():
    # st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Date Based Lookup", "Overall Analytics","Training Data Lookup"])
    
    if app_mode == "Date Based Lookup":
        source_lookup()
    elif app_mode == "Overall Analytics":
        st.code(open('labeling_dashboard.py').read())
    elif app_mode=='Training Data Lookup':
        training_data_lookup()
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