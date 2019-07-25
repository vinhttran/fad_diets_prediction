def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False

def keyword_column_boolean(df, keyword_list):
    for x in keyword_list:
        df[x] = df['text'].apply(lambda text: word_in_text(x,text))
# sentiment analysis
def get_tweet_polarity(tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(tweet)
        # set sentiment
        return analysis.sentiment.polarity


def get_tweet_sentiment(polarity):
        if polarity > 0:
            return 'positive'
        elif polarity == 0:
            return 'neutral'
        else:
            return 'negative'
def load_US_coord_dict():
    '''
    Input: n/a
    Output: A dictionary whose keys are the location names ('City, State') of the
    378 US classification locations and the values are the centroids for those locations
    (latitude, longittude)
    '''

    pkl_file = open("GeoData/US_coord_dict.pkl", 'rb')
    US_coord_dict = pickle.load(pkl_file)
    pkl_file.close()
    return US_coord_dict

def find_dist_between(tup1, tup2):
    '''
    INPUT: Two tuples of latitude, longitude coordinates pairs for two cities
    OUTPUT: The distance between the cities
    '''

    return np.sqrt((tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2)

def closest_major_city(tup):
    '''
    INPUT: A tuple of the centroid coordinates for the tweet to remap to the closest major city
    OUTPUT: String, 'City, State', of the city in the dictionary 'coord_dict' that is closest to the input city
    '''

    d={}
    for key, value in US_coord_dict.items():
        dist = find_dist_between(tup, value)
        if key not in d:
            d[key] = dist
    return min(d, key=d.get)

def get_closest_major_city_for_US(row):
    '''
    Helper function to return the closest major city for US users only. For users
    outside the US it returns 'NOT_IN_US, NONE'
    '''
    return closest_major_city(row['coordinate_point'])
