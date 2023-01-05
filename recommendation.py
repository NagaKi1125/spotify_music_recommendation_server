import os

import lyricsgenius
import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from yellowbrick.target import FeatureCorrelation
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
import joblib
import warnings

warnings.filterwarnings("ignore")


def get_decade(year):
    period_start = int(year / 10) * 10
    decade = '{}s'.format(period_start)
    return decade


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


class MusicRecommendation:
    # read data
    data = pd.read_csv("data/data.csv")
    genre_data = pd.read_csv('data/data_by_genres.csv')
    year_data = pd.read_csv('data/data_by_year.csv')
    lyrics_df = pd.read_csv('data/lyrics.csv')
    # spotify
    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(client_id="44d387e458e547b3a49f511e3536c568",
                                              client_secret="412f731ca8e94ed7b0a5d5dbefc1886c"))
    genius = lyricsgenius.Genius('gJNyqDyWEN_mcJJg20C_LALYW9ZpKEI4tWdhWLoIK6duEUTrRadLX8S8GOLMZO4d')

    # feature
    feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                     'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'duration_ms', 'explicit',
                     'key',
                     'mode',
                     'year']
    # Create a list of the feature names
    features = np.array(feature_names)
    sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']

    def __int__(self, *args, **kwargs):
        super(MusicRecommendation, self).__init__(self)

    def find_song(self, name, year):
        song_data = defaultdict()
        results = MusicRecommendation.sp.search(q='track: {} year: {}'.format(name, year), limit=1)
        if not results['tracks']['items']:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = self.sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)

    def get_song_data(self, song, spotify_data):
        try:
            song_data = spotify_data[(spotify_data['name'] == song['name'])
                                     & (spotify_data['year'] == song['year'])].iloc[0]
            return song_data

        except IndexError:
            return self.find_song(song['name'], song['year'])

    def get_mean_vector(self, song_list, spotify_data, number_cols):
        song_vectors = []

        for song in song_list:
            song_data = self.get_song_data(song, spotify_data)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                continue
            song_vector = song_data[number_cols].values
            song_vectors.append(song_vector)

        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

    def recommend_songs(self, song_list, number_cols, song_cluster_pipeline, n_songs=10):
        metadata_cols = ['name', 'year', 'artists', 'id']
        song_dict = flatten_dict_list(song_list)

        song_center = self.get_mean_vector(song_list, self.data, number_cols=number_cols)
        scaler = song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(self.data[number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])

        rec_songs = self.data.iloc[index]
        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')

    def main(self, seedList, song_cluster_pipeline):
        # save the model
        # song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
        #                                   ('kmeans', KMeans(n_clusters=20,
        #                                                     verbose=False))
        #                                   ], verbose=False)
        #
        # dtypes = MusicRecommendation.data.select_dtypes(np.number)
        # song_cluster_pipeline.fit(dtypes)
        # file_name = 'model.joblib'
        # joblib.dump(song_cluster_pipeline, file_name)

        # load the cluster pipeline model to better use
        # song_cluster_pipeline = joblib.load(file_name)

        number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
                       'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

        recommendation = self.recommend_songs(song_list=seedList, song_cluster_pipeline=song_cluster_pipeline,
                                              number_cols=number_cols)

        return recommendation
        # return 'hello world'

    # DATA ANALYZE
    def getArtists(self):
        artists_list = []
        for index, row in self.data.iterrows():
            artists_array = row['artists']
            artists_array = artists_array.replace('[', '').replace(']', '').replace("'", '')
            if "," in artists_array:
                artist = artists_array.split(", ")
                for a in artist:
                    artists_list.append(a)
            else:
                artists_list.append(artists_array)
        artists_list = list(dict.fromkeys(artists_list))
        artists_response = []
        index = 0
        for ar in artists_list:
            if index <= 200:
                result = MusicRecommendation.sp.search(q='{}'.format(ar), type='artist', limit=1)
                items = result['artists']['items']
                # get artist infor
                if items:
                    artist_info = items[0]

                    # get artist thumbnail url
                    url = ''
                    if artist_info['images']:
                        url = artist_info['images'][0]['url']
                    artist = {'spotify_id': items[0]['id'],
                              'thumbnail': url,
                              'name': artist_info['name'],
                              'popularity': artist_info['popularity']}

                    artists_response.append(artist)
            else:
                break
            index += 1
        return artists_response

    def getTrackInformation(self, trackId, year):
        track_info = MusicRecommendation.sp.track(track_id=trackId)

        spotify_id = trackId
        name = track_info['name']
        artists = []
        preview_url = ''
        duration = ''
        preview_url = track_info['preview_url']
        duration = track_info['duration_ms']

        if track_info['artists']:
            for a in track_info['artists']:
                artist = MusicRecommendation.sp.artist(artist_id=a['id'])
                url = ''
                if artist['images']:
                    for artist_thumbnail in artist['images']:
                        url = artist_thumbnail['url']
                        break

                ar = {'spotify_id': artist['id'],
                      'thumbnail': url,
                      'name': artist['name'],
                      'popularity': artist['popularity']}
                artists.append(ar)

        song = {
            'spotify_id': spotify_id,
            'year': year,
            'name': name,
            'artists': artists,
            'preview_url': preview_url,
            'duration': duration
        }

        return song

    def fetchTrack(self):
        songs = []
        songList = MusicRecommendation.data[['id', 'name', 'year']]
        for index, s in songList.iterrows():
            if index <= 100:
                song = self.getTrackInformation(trackId=s['id'], year=s['year'])
                songs.append(song)
            else:
                break
        return songs

    def getTrackLyrics(self, name):
        lyric = ''
        for index, l in MusicRecommendation.lyrics_df.iterrows():
            if name == l['song']:
                lyric = {'track_name': l['song'], 'lyric': l['seq']}
                break
        return lyric

