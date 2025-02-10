# -*- coding: utf-8 -*-
"""

# **Analyzing music behaviour and predicting playlists**

Ideen:


*   three songs -> new ideas
*   load, preprocess data

*   streamlit app as input formular for extra criteria


  *   visualization and analysation
      *   features over the years (bmp, dancability, energy, valence, ...)
      *   bpm, dancability and valence (3D)
      * average length over the years (statistical analysis)
      *   bpm, dancability, liveness
      * distribution of key features (e.g., energy, danceability, acousticness) across different languages

* ML
  * clustering into genres (k means oder hierarchical clustering)
    * clustering using multiple audio features (using K-Means or Hierarchical Clustering) to identify different "types" of tracks that might not necessarily align with traditional genres
  * song prediction to 3 given songs (k nearest neighbors?)

Requirements to have installed:
 * kaggle
 * numpy  ?preinstalled
 * pandas ?preinstalled

# **Data description**
"""
#import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import streamlit as st

def streamlit_setup():
  st.set_page_config(
      page_title="Playlist Prediction",
      page_icon="♫"
  )
  # display title and short explanation
  st.title("Welcome to our Playlist Prediction")
  st.write("where you might discover your new favorite songs!")

def load_dataframe(link):
# Download latest version
  #path = kagglehub.dataset_download(link)
  path = "spotify_tracks.csv"
  #print("Path to dataset files:", path)

  # Load the dataset
  #df = pd.read_csv(path+'/spotify_tracks.csv')
  df = pd.read_csv(path)

  # Display the first few rows
  df.head()
  return df

def preprocess_data(df):
  df = df.drop_duplicates(subset=['track_name','artist_name']) #deletes all row duplicates
  df = df.drop(['liveness','artwork_url','mode'], axis = 1) #delete the columns liveness, artwork_url and mode
  df = df[~(df == -1).any(axis=1)] #deletes all entries with values that are -1 (not existent)

  # substitute track-id values by numbers from 0 to len(dataframe)
  df['track_id'] = range(len(df))
  df.head()
  return df

streamlit_setup()
df = load_dataframe("https://www.kaggle.com/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week")
df = preprocess_data(df)
# **Data analysis and visualization**

## 1) General data analysis
# df.describe()
st.dataframe(df.describe(), use_container_width=True)

#creating a histogram for song releases over the years
with st.expander("General Data Analysis"):
  fig = plt.figure()
  gs = fig.add_gridspec(2, 2)
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[0, 1])
  ax4 = fig.add_subplot(gs[1, 1])

  ax1.hist(df['year'], bins = 50, color = "maroon", density = "True")
  ax1.set_xlabel("year")
  ax1.set_title("Song releases over the years")

  ax2.hist(df['danceability'], bins ='auto', color = "#D2665A",density = "True")
  ax2.set_xlabel("danceability")
  ax2.set_title("Distribution of danceability")

  ax3.hist(df['instrumentalness'], bins = 30, color = "#F2B28C",density = "True")
  ax3.set_xlabel("instrumentalness")
  ax3.set_title("Distribution of instrumentalness")

  ax4.hist(df['duration_ms'], bins = 50, color = "#F6DED8",density = "True")
  ax4.set_xlabel("song length in ms")
  ax4.set_title("Distribution of song length")
  fig.tight_layout()
  #plt.show()
  st.pyplot(fig)


"""From the plotted graphs, we are able to draw the following conclusions:

-the number of songs which were released each year increased over the last 50 years. Last year, the amount of released music was much higher than before.

-Most of the songs have a dancability score between 0.4 and 0.8

-Nearly all songs have a very low instrumentalness score and are rather short.

## 2) Exploration of cultural patterns and trends
"""
with st.expander("Cultural Patterns and Trends"):
  print("to do")


"""## 3) Correlation between features"""
with st.expander("Correlation between features"):
  fig, ax = plt.subplots()
  ax.scatter(x=df["danceability"]*100,y=df["popularity"],alpha = 0.09, color = "#7C444F")
  ax.set_xlabel("dancability in %")
  ax.set_ylabel("popularity in %")
  ax.set_title("Relation beween danceability and acousticness")
  ax.grid(True)
  fig.tight_layout()
  # plt.show()
  st.pyplot(fig)

"""As the plot displays, the most popular songs have a medium or high danceability.
Nevertheless, it is likely that the popularity does not only depend on the dancability.
"""



"""
#**Playlist prediction**"""
def contrast_coding(df, column_name):
    df[column_name] = df[column_name].astype('category')
    df[column_name] = df[column_name].cat.codes
    return df
# contrast coding the language column
df = contrast_coding(df, 'language')

# select relevant features for KNN
features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'language']
X = df[features].values

# Initialize and fit the KNN model
knn = NearestNeighbors(n_neighbors=13, algorithm='auto', metric='cosine')
knn.fit(X)

def recommend_songs(knn, song_ids):
    if not song_ids:
        return pd.DataFrame()

    # Find average feature vector for the input songs
    #avg_features = np.mean(X[song_ids], axis=0)
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'language']
    avg_features = np.mean(df.loc[song_ids, features], axis=0)

    # Find k-nearest neighbors to average vector
    distances, ids = knn.kneighbors([avg_features], n_neighbors=13)

    # Return recommended song indices (excluding the input songs)
    recommended_ids = [i for i in ids[0] if i not in song_ids]
    recommended_songs = df.loc[recommended_ids, ['track_name', 'artist_name', 'year']]
    return recommended_songs


# Example usage in an interactive loop
with st.expander("Playlist Prediction"):
  selected_songs = []
  for i in range(3):
      search_term = st.text_input(f"Enter the name of song {i+1}: ", "Shape of You")
      matching_songs = df[df['track_name'].str.contains(search_term, case=False)]

      if matching_songs.empty:
          st.write("No matching songs found. Please try again.")
          continue

      # show options in st.multiselect in form Id: Song Name by artist in year and saves id in variable
      selected_song_name = st.multiselect(
        
        label="Select one of the songs",
        options=matching_songs['track_name'].values,
        max_selections=1
        )[0]
      
      
            #returning the index of selected_song_name in df
      selected_song_id = df[df["track_name"] == selected_song_name].index.values[0]
      st.write("Selected song: ", selected_song_id, selected_song_name)

      selected_songs.append(selected_song_id)

      #st.write(selected_song_name)

  recommended_songs_df = recommend_songs(knn, selected_songs)

  if not recommended_songs_df.empty:
      print("\nRecommended Playlist:")
      print(recommended_songs_df)
  else:
      print("No recommendations found for this song.")




