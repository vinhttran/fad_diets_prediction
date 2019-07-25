import json
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap

def prevalence(df):
    sns.set_context("paper", font_scale=2)
    fig, ax = plt.subplots(figsize = (16,8))
    ax = sns.distplot(df["OBESITY_AdjPrev"], color = 'seagreen');
    ax.set(xlabel="Prevalence of Obesity",
           ylabel = "Percentage of Cities")
    plt.axvline(np.mean(df["OBESITY_AdjPrev"]), 0,6, color = "seagreen", label = "Mean")
    plt.axvline(15.3, 0,6, color = "steelblue",linestyle='dashed', label="Low cut-off")
    plt.axvline(26.8, 0,6, color = "crimson",linestyle='dashed', label="Medium cut-off")
    plt.axvline(32.8, 0,6, color = "darkviolet",linestyle='dashed', label="High cut-off")

    plt.legend()

    ax.axes.set_title("Obesity Prevalance in US Cities",fontsize=30);


def generateBaseMap(default_location=[29.4167, -98.5], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


# read in data
df = pd.read_json('.txt', lines = True)

#fix coordinates field, fill None with 0
filled = df['coordinates'].fillna(0)
#replace old column
df["coordinates"] = filled
#mask
coordinates = df[df['coordinates'] != 0]

#keep only needed vars;
coordinates = coordinates[["text","coordinates", "created_at", "lang", "possibly_sensitive", "id_str"]].reset_index()

#create separate lon and lat columns
coordinates_pd = pd.DataFrame(list(np.array(coordinates["coordinates"])))
coordinates_pd.rename(columns={'coordinates': 'coordinate_point'}, inplace=True)
df_coordinates = pd.concat([coordinates, coordinates_pd], axis=1, join='inner')
df_coordinates[['lon', 'lat']] = pd.DataFrame(df_coordinates['coordinate_point'].tolist())


base_map = generateBaseMap()

df_coordinates["count"] = 1
HeatMap(data=df_coordinates[['lat', 'lon', 'count']].groupby(['lat', 'lon']).sum().reset_index().values.tolist(), radius=8, max_zoom=13).add_to(base_map)

base_map
