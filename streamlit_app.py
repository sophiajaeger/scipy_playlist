"""
Analyzing music behaviour and predicting playlists

data source: https://www.kaggle.com/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week (downloaded: 10th February 2025)

note for grading: some songs' language is not listed correctly in the dataset. This impacts the output of the recommendations when filtering by lamguage.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def load_dataframe():
  path = "spotify_tracks.csv"
  df = pd.read_csv(path)
  return df

def preprocess_data(df):
  df = df.drop_duplicates(subset=['track_name','artist_name']) #deletes all row duplicates
  df = df.drop(['liveness','artwork_url','mode'], axis = 1) #delete the columns liveness, artwork_url and mode
  df = df[~(df == -1).any(axis=1)] #deletes all entries with values that are -1 (not existent)

  # rename track_id to spotify_id
  df = df.rename(columns={'track_id': 'spotify_id'})
  
  # generate a new column track_id with values from 0 to len(dataframe)
  df['track_id'] = range(len(df))

  # set the track_id as the DataFrame index to standardize indexing
  df.set_index('track_id', inplace=True)
  return df

st.set_page_config(
    page_title="Playlist Prediction",
    page_icon="♫"
)
st.title("Welcome to our Playlist Prediction")
st.write("where you might discover your new favorite songs!")

st.write("This app is based on the dataset from 'https://www.kaggle.com/gauthamvijayaraj/spotify-tracks-dataset-updated-every-week', downloaded on 10th February 2025")
df = load_dataframe()
df = preprocess_data(df)
with st.expander("Data Description"):
    st.write("The columns include features such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, valence, tempo, and popularity. The dataset also contains information about the track name, artist name, year, language, and the Spotify ID of the track.")
    st.write("The dataset contains", df.shape[0], "rows.")
    st.write("A few examples of the data are displayed below:")
    st.dataframe(df.head())

st.markdown("## Data analysis and visualization")

st.markdown("### 1) General data analysis")
with st.expander("General Data Analysis"):
  st.dataframe(df.describe(), use_container_width=True)
  st.write("This table gives us insights about the maximum and minimum values of a parameter, as well as statistical parameters like the mean which help to process the data correctly and use the parameters in a good manner for further analysis.")

with st.expander("General visualization"):
  # show distribution of languages as a pie chart
  st.write("\n \n In the following pie chart one can see the distribution of songs across different languages.")
  cola, colb = st.columns(2)
  with cola:
    fig, ax = plt.subplots()
    df['language'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
    ax.set_ylabel('')
    ax.set_title('Distribution of Songs by Language')
    #plt.show()
    st.pyplot(fig)
  with colb:
    st.write("The pie chart shows that a large share of songs in the dataset are in English, followed by Tamil and those with an unknown language. The other languages have a much smaller share of the dataset.")

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

  st.text("From the plotted graphs, we are able to draw the following conclusions:\n - the number of songs which were released each year increased over the last 50 years. Last year, the amount of released music was much higher than before. This might be due to modern technologies enabling a larger amount of people to publish their own music.\n - Most of the songs have a dancability score between 0.4 and 0.8 \n - Nearly all songs have a very low instrumentalness score and are rather short. A low instumentalness score might be the result of nearly all songs having lyrics. \n - Most of the songs have a length shorter than two minutes. This could be the case because white noise tracks or other are included")
  
st.markdown("### 2) Exploration of Cultural Patterns and Trends")

# plot for patterns over the years
with st.expander("Development of patterns over the year"):
  fig, ax1 = plt.subplots(figsize=(12, 8))
  sns.lineplot(data=df, x='year', y=df['tempo'], marker='o', label='Tempo', color='#D2665A', ax=ax1)
  ax1.set_ylabel('Tempo', color='#D2665A')
  ax1.tick_params(axis='y', labelcolor='#D2665A')
  ax2 = ax1.twinx()
  sns.lineplot(data=df, x='year', y=df['valence'] * 100, marker='o', label='Valence (*100)', color='#F2B28C', ax=ax2)
  ax2.set_ylabel('Valence (*100)', color='#F2B28C')
  ax2.tick_params(axis='y', labelcolor='#F2B28C')
  sns.lineplot(data=df, x='year', y=df['duration_ms'] / 6000, marker='o', label='Length (seconds)', color='#F6DED8', ax=ax2)
  ax1.set_xlabel('Year')
  plt.title('Music Attributes Over Years')
  # Show the plot
  #plt.show()
  st.pyplot(fig)
  st.text("In this plot, the distribution of the features Length of the song, Tempo and Valence can be investigated. We implemented two axis to integrate all three plots into one figure. Tempo is plotted with the use of the left axis while Valence and Length of the song are plotted with the help of the right axis. Especially for the parameter Valence, we can recognize strong changes over the years, with an overall decreasing tendency during the last 30 years. The same is true for Length, while the parameter tempo stays more or less constant. ")

# check if there are statistically significant differences in the distribution of key features across languages using mixed linear effects models
with st.expander("Differences of features across languages"):
    features = ['tempo', 'danceability', 'speechiness', 'energy']
    colors = ["#AEC6CF", "#FFB347", "#77DD77", "#F49AC2"]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    posthoc_results = {}
    for i, feature in enumerate(features):
        # Plot distribution of the feature across languages using a boxplot
        sns.boxplot(x='language', y=feature, data=df, ax=axs[i], color=colors[i])
        axs[i].set_title(f"{feature.capitalize()} by Language")
        axs[i].set_xlabel("Language")
        axs[i].set_ylabel(feature.capitalize())
        
        # Fit a linear model (ANOVA) with language as categorical
        model = smf.ols(formula=f"{feature} ~ C(language)", data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_val = anova_table.loc["C(language)", "PR(>F)"]
        
        # Perform pairwise post-hoc Tukey HSD test to identify pairwise differences
        tukey = pairwise_tukeyhsd(endog=df[feature], groups=df['language'], alpha=0.05)
        # Convert Tukey results to a DataFrame
        tukey_results = pd.DataFrame(data=tukey._results_table.data[1:], 
                                     columns=tukey._results_table.data[0])
        # Filter for comparisons that are statistically significant
        sig_comparisons = tukey_results[tukey_results['reject'] == True]
        posthoc_results[feature] = sig_comparisons
        
    fig.tight_layout()
    st.pyplot(fig)
    st.write("Here we can investigate the expression of different features (Tempo, Danceability, Speechiness and Energy) in the six different languages of the dataset. Below, the results of a pairwise comparison in form of the Tukey HSD Test are displayed for each feature. It has to be mentioned that only the pairs which lead to significant results are listed here. Interestingly, the significant differences are very high for the parameter Tempo (e.g Hindi/Korean or English/Korean) in comparison to the other parameters. There are more significant differences for the parameters Danceablity and Speechiness in contrast to Tempo and Energy. This leads to the conclusion that there are more inter-linguistic differences in the parameters Danceability and Speechiness opposed to Tempo and Energy.")
    
    st.markdown("#### Pairwise Comparisons (Tukey HSD Test)")
    st.write("We will use the Tukey HSD test to identify which languages have statistically significantly different feature values.")

    summary = summary = {
      "tempo": "English and Hindi songs tend to have a faster tempo than Korean tracks by about 6 bpm and Tamil by about 3 bmp. This may reflect a cultural preference for faster-paced music, possibly to encourage dancing, whereas the other cultures may favour a more relaxed approach.",
      "danceability": "English songs consistently show higher danceability scores than all other language groups. This may reflect a cultural emphasis in English-language music on producing songs that are particularly engaging and suitable for dancing.",
      "speechiness": "Hindi songs exhibit a higher level of speechiness compared to English, Korean, and Unknown. This higher speechiness hints at a cultural affinity for lyrical storytelling or rap elements",
      "energy": "English songs show a significantly higher energy compared to all other groups (by up to 0.28 when compared with Korean songs), reflecting a dynamic and vigorous style. This reflects the modern pop production standards especially prominent in genres like pop and rock. Korean music, on the other hand, are among the least energetic compared to other languages, perhaps focusing more on melody or atmospheric production."
    }
    checked = st.checkbox("I want to see the complete results of the Tukey HSD test")
    for feature in features:
        st.markdown(f"**{feature.capitalize()}**")
        st.write(summary[feature])
        if posthoc_results[feature].empty:
            st.write("However, no statistically significant differences were found between languages for this feature.")
        else:
            if checked:
              #show the collumns "group1", "group2", "meandiff", "p-adj" of the dataframe posthoc_results[feature]
              st.dataframe(posthoc_results[feature][["group1", "group2", "meandiff", "p-adj"]])
            else:
              st.write("Click the checkbox above to see the complete results.")

st.markdown("### 3) Correlation between features")
with st.expander("Dancability and acousticness"):
  fig, ax = plt.subplots()
  ax.scatter(x=df["danceability"]*100,y=df["popularity"],alpha = 0.09, color = "#7C444F")
  ax.set_xlabel("dancability in %")
  ax.set_ylabel("popularity in %")
  ax.set_title("Relation beween danceability and acousticness")
  ax.grid(True)
  fig.tight_layout()
  # plt.show()
  st.pyplot(fig)

  st.text("As the plot displays, the most popular songs have a medium or high danceability.\n Nevertheless, it is likely that the popularity does not only depend on the dancability.\n")
# plot to visualize song length
with st.expander("Song length per year"):
  fig, ax = plt.subplots()
  ax.scatter(x=df["year"],y=df["duration_ms"],alpha = 0.9, color = "#7C444F")
  ax.set_xlabel("year")
  ax.set_ylabel("song length")
  ax.set_title("Relation beween release year and song length")
  ax.grid(True)
  fig.tight_layout()
  # plt.show()
  st.pyplot(fig)
  st.write("There is an interesting correlation between year and the length of songs released in that year. In the earlier years of the dataset, from around 1970 until 1990, the song length was relatively stable. After that a few songs had an increased length, with the maximum song length per year almost following an exponential curve between 2000 and 2020. However, it declined very fast after 2020. However, the mayority of songs released throughout all these years has a duration of less than 1*10^6 milliseconds, which is equivalent to 100 seconds with more songs having a length between 100 and 200 seconds after 1995.")
with st.expander("Heatmap of feature correlations"):
  # Create a correlation matrix
  df_numerical = df.select_dtypes(include=[np.number])
  corr = df_numerical.corr().round(2)
  
  # Create a heatmap
  plt.figure(figsize=(10, 8))
  sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
  
  # Set plot title
  plt.title('Correlation Matrix')
  
  # Show the plot
  #plt.show()
  st.pyplot(plt)
  st.text("With the help of this heatmap, we are able to infer which parameters in the data might me correlated and therefore interesting to further investigate. \n 1)Loudness/Energy has a high positive correlation (0.82).  The higher the volume of the song, the more energetic the is perceived to be. \n 2) Valence/Danceability has a high positive correlation (0.64). The happier the song makes a person feel, the more the person is motivated to dance to it. \n 3) Acousticness/Energy has a high negative correlation (-0.64). It seems to be the case that songs that are acoustic are perceived as less energetic \n 4)Tempo/Danceability has almost no correlation at all (0.07), indicating that the tempo of the song does not influence the ability to dance to it. \n 5) The key the song is written in does not have an influence on valence (correlation=0.07), which is kind of suprising when we think about having major and minor chords which are normally linked to the emotional character of a song.")

def contrast_coding(df, column_name):
    # contrast code a categorical column but keep the original column
    df[column_name + '_original'] = df[column_name]
    df[column_name] = df[column_name].astype('category')
    df[column_name] = df[column_name].cat.codes
    return df

# contrast coding the language column
df = contrast_coding(df, 'language')

def fit_knn(df):
  # select relevant features for KNN
  features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'language']
  X = df[features].values

  # initialize and fit the KNN model
  knn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
  knn.fit(X)
  return knn

def recommend_songs(knn, filtered_df, song_ids):
    if not song_ids:
        return pd.DataFrame()

    # compute average feature vector of input songs
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'language']
    avg_features = np.mean(df.loc[song_ids, features], axis=0)

    # k-nearest neighbors to average vector
    neighbor_positions = knn.kneighbors([avg_features], n_neighbors=10, return_distance=False)
    recommended_ids = filtered_df.iloc[neighbor_positions.flatten()].index.tolist()
    return recommended_ids

def label_preview(row):
    return f"{row['artist_name']} ({row['year']}): {row['track_name']}"
  
st.markdown("## Playlist Prediction")
st.write("Please enter the names of three songs to get recommendations.")

selected_songs = [None, None, None]
for i in range(3):
  search_term = st.text_input(f"Enter the name of song {i+1}: ")
  
  # create a dataframe 'matching songs' with all songs where the track_name contains the search term
  matching_songs = df[df['track_name'].str.contains(search_term, case=False)]
  
  if matching_songs.empty:
      st.write("No matching songs found. Please try again.")
      continue
  
  # create a new column named preview that concatinates the columns "track_title" by "artist_name" in "year"
  matching_songs['preview'] = matching_songs.apply(label_preview, axis=1)

  # show options in st.multiselect
  selected_preview = st.selectbox(
    label="Select one of the songs",
    options=matching_songs['preview'],
    key=f"song_{i}",
    placeholder="Select a song..."
    )

  # show the selection to the user 
  selected_song_id = matching_songs[matching_songs['preview'] == selected_preview].index[0]
  st.write("Selected song: ", selected_song_id, selected_preview)

  selected_songs[i] = selected_song_id

# check if 3 different songs were selected
if None in selected_songs:
  st.write("Please select three songs to continue.")
elif len(set(selected_songs)) != 3:
    st.write("Please select three different songs.")
else:
    # create a  form to select filters for the recommendations
    with st.form(key='preselection_form'):
      years = st.slider("Select the years of the recommended songs", min_value=min(df['year']), max_value=max(df['year']), step=1, value=[min(df['year']), max(df['year'])])
      languages = st.multiselect("Select the languages of the recommended songs", df['language_original'].unique())

      submitted = st.form_submit_button("Generate Playlist")

    if submitted:
      # filter the dataframe based on the selected years and languages
      filtered_df = df[(df['year'] >= years[0]) & 
                       (df['year'] <= years[1]) & 
                       (df['language_original'].isin(languages))]
      # fit the knn model
      knn = fit_knn(filtered_df)
      # get the recommended song ids
      recommended_song_ids = recommend_songs(knn, filtered_df, selected_songs)

      # create a new dataframe with the recommended songs
      recommended_songs_df = df.loc[recommended_song_ids]

      if not recommended_songs_df.empty:
        #generate playlist
        # extract spotify ids from the recommended songs
        spotify_ids = recommended_songs_df['spotify_id'].tolist()

        col1, col2 = st.columns(2)
        # display every other in the second column with embedded spotify player
        for i, spotify_id in enumerate(spotify_ids):
          embed_code = f"""
          <iframe src="https://open.spotify.com/embed/track/{spotify_id}" 
                  width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media">
          </iframe>
          """
          if i % 2 == 0:
            with col1:
              st.components.v1.html(embed_code, height=90)
          else:
            with col2:
              st.components.v1.html(embed_code, height=90)
        # print the rows of the recommended songs
        with st.expander("View the full dataframe of the recommendations"):
          st.dataframe(recommended_songs_df)
