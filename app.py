from recommendation import MusicRecommendation
from flask import Flask, redirect, url_for, request
import joblib

app = Flask(__name__)

music_recommend = MusicRecommendation()
file_name = 'model.joblib'
song_cluster_pipeline = joblib.load(file_name)


@app.route('/recommend', methods=['GET'])
def recommendation():
    seedList = request.get_json()
    result = music_recommend.main(seedList=seedList, song_cluster_pipeline=song_cluster_pipeline)

    return result


@app.route('/auto-gen/fetch-track', methods=['GET'])
def fetch_track():
    result = music_recommend.fetchTrack()
    return result


# @app.route('/track/lyric', methods=['GET'])
# def getTrackLyric():
#     track_name = request.get_data()
#     result = music_recommend.getTrackLyrics(name=track_name)
#     return result


# data analyze
@app.route('/auto-gen/artists', methods=['GET'])
def get_artists():
    artists = music_recommend.getArtists()
    return artists


@app.route('/auto-gen/getTrack', methods=['GET'])
def get_tracks():
    seedList = request.get_json()
    music_list = []
    for r in seedList:
        track = music_recommend.getTrackInformation(trackId=r['id'], year=r['year'])
        music_list.append(track)
    return music_list


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run(threaded=True)
