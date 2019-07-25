import pandas as pd

def clean_census(census_df):
    #values too low are set to zero as indicated by letters
    df2 = df.replace({'D': 0, 'F':0, 'FN':0, 'NA':0, 'S':0, 'X':0, 'Z':0})

    #remove % sign to cast as floats
    df3 = df2.replace('%','',regex=True)

    # fix city and state names and convert to abbreviation
    city = df3['city_state'].str.split(',', expand=True)
    city.rename(columns = {0:"closest_city", 1: "closest_state"}, inplace=True)
    #manually fix some city names for merge
    city['closest_city'] = city['closest_city'].str.replace(' city', '')
    city['closest_city'] = city['closest_city'].str.replace(' town', '')
    city['closest_city'] = city['closest_city'].str.replace(' CDP', '')
    city['closest_city'] = city['closest_city'].str.replace(' county', '')
    city['closest_city'] = city['closest_city'].str.replace(' County', '')
    city['closest_city'] = city['closest_city'].str.replace(' municipality', '')
    city['closest_city'] = city['closest_city'].replace("Baltimore Highlands", 'Baltimore')
    city['closest_city'] = city['closest_city'].replace("Indianapolis (balance)", 'Indianapolis')
    city['closest_city'] = city['closest_city'].replace("Nashville-Davidson (balance)", 'Nashville')
    city['closest_city'] = city['closest_city'].replace("Augusta-Richmond (balance)", 'Augusta')
    city['closest_city'] = city['closest_city'].replace("Louisville/Jefferson (balance)", 'Louisville')
    city['closest_city'] = city['closest_city'].replace("Athens-Clarke (balance)", 'Louisville')

    #manually fix some state names for merge
    city['closest_state'] = city['closest_state'].str.replace("Fact", '')
    city['closest_state'] = city['closest_state'].str.lstrip(" ")
    city['closest_state'] = city['closest_state'].replace("Virginia (County)", 'Virginia')

    #convert state name to abbreviation
    us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
    }

    #replace state name with state abbreviation
    city["closest_state"].replace(us_state_abbrev, inplace=True)

    #remove city_state because it is not numerical
    numbers = df3.drop(["city_state"], axis=1)

    #remove commas
    numbers2 = numbers.replace(',','',regex=True)

    #actual number casting
    cols = numbers2.select_dtypes(exclude=['float']).columns
    numbers2[cols] = numbers2[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

    #set city and state df back to the numbers
    df4 = pd.concat([city, numbers2], axis=1, join='inner')

    #two exact duplicates
    df5 = df4.drop_duplicates()

    #output to pickle file

    pd.to_pickle(df5, 'data/census_data.pkl')
