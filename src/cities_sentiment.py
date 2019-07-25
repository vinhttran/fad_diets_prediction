import json
import pandas as pd
import matplotlib.pyplot as plt
import re
import multiprocessing
import numpy as np
import shutil
import preprocessor as p
import pickle
from textblob import TextBlob
import itertools

%matplotlib inline
plt.style.use('ggplot')

pd.set_option('display.max_columns', 500)
pd.options.display.max_rows = 500

#run in order

def clean_one(df):
    '''initial round of dataframe cleaning'''

    #only keep variables needed
    df = df[["id_str", "text", "place", "coordinates", "created_at", "lang", "possibly_sensitive","retweeted" ]]

    #don't include retweets
    keyword_column_boolean(df, ['RT'])
    df = df[df['RT']==False]

    #booleans for each diet
    diet_list = ['keto','whole30','gluten','mediterranean','lowfat', 'atkins', 'paleo', 'celeryjuice']
    keyword_column_boolean(df, diet_list)

    #fix Place field, first fill None with 0
    filled = df['place'].fillna(0)

    #replace old column
    df["place"] = filled

    # #mask
    place = df[df['place'] != 0]
    place.reset_index(inplace=True)

    #pull out bounding box from place
    df_place = [i for i in df["place"] if i]
    df_place_2 = pd.DataFrame(list(np.array(df_place)))

    city = df_place_2[["name","country_code"]]

    bounding_box = pd.DataFrame(list(np.array(df_place_2["bounding_box"])))
    bounding_box.rename(columns={'coordinates': 'bounding_box'}, inplace=True)

    #add to original df to get location
    df_location = pd.concat([place, bounding_box], axis=1, join='inner')
    df_location = pd.concat([df_location, city], axis=1, join='inner')

    #limit to only english for analysis
    df_location = df_location[df_location['lang'] == 'en']

    df_eng = df[df['lang'] == 'en']

    #limit to only US for analysis
    df_location = df_location[df_location['country_code'] == 'US']

    #remove this one id
    df_location = df_location[df_location["id_str"] != 1141697585700204544]

    #fix index
    df_location2 = df_location.reset_index()
    df_location3 = df_location2.drop(["index", "level_0"], axis = 1)
    df_location3['index'] = df_location3.index

    #final df
    df_clean = df_location3

    return df_clean


def clean_text(df_clean):
    '''#clean tweets - remove URLs, smileys, mentions, emojis'''
    p.set_options(p.OPT.URL, p.OPT.SMILEY, p.OPT.MENTION, p.OPT.EMOJI)

    text_list = list(df_clean["text"])
    clean_text_list = []

    for tweet in text_list:
        clean_text_list.append(p.clean(tweet))

    df_clean["text_clean"] = clean_text_list

    return df_clean


def sentiment(df_clean):
    '''get sentiment of tweet'''
    df_clean = df_clean[['text_clean', "index"]]

    # using TextBlob calculate polarity and sentiment on clean tweets
    df_clean['polarity'] = df_clean['text_clean'].map(get_tweet_polarity);
    df_clean['sentiment'] = df_clean['polarity'].map(get_tweet_sentiment);

    return df_clean



def map_city(df_clean):
    '''get coordinates from place variable'''
    list2 = [item[0] for item in df_clean["bounding_box"]]
    list3 = [item[0] for item in list2]
    list4 = [item[::-1] for item in list3]

    df_clean["coordinate_point"] = list4

    if __name__ == "__main__":

        # Load US_coord_dict
        US_coord_dict = load_US_coord_dict()

        # Create a new column called 'closest_major_city'
        df_clean['closest_major_city'] = df_clean.apply(lambda row: get_closest_major_city_for_US(row), axis=1)

        prediction = df_clean

        return prediction


def clean_two(prediction):
    '''more dataframe cleaning'''
    #aggregate for not binned data by closest_major_city
    prediction2 = prediction.groupby(['closest_major_city','sentiment'])['polarity'].mean()

    prediction2 = pd.DataFrame(prediction2)
    prediction3 = prediction2.pivot_table(index='closest_major_city', columns="sentiment", values='polarity')
    prediction3 =  prediction3.rename_axis(None, axis=1).reset_index()
    prediction4 = prediction3.rename(columns = {"index": 'PlaceName'})

    #pull out city and state into separate columns
    city = prediction4["closest_major_city"].str.split(',', expand=True)
    city.rename(columns = {0:"closest_city", 1: "closest_state"}, inplace=True)
    prediction5 = pd.concat([city, prediction4], axis=1, join='inner')

    #remove spaces for merging
    prediction5["closest_state"] = prediction5["closest_state"].str.strip()

    return prediction5


def cdc_cut(cities):

    '''bin CDC obesity rates for classifier, putting into 3 bins '''

    cities["OBESITY_cut"] = pd.qcut(cities["OBESITY_AdjPrev"],3,
                                    labels = ["low", "medium", 'high'])
    return cities
