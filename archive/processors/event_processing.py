import pandas as pd
from utils.data_utils import preprocess_data

def process_events(news_data, conflict_data):
    news_df = preprocess_data(news_data)
    conflict_df = preprocess_data(conflict_data)

    events = pd.concat([news_df, conflict_df])
    return events
