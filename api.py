from flask import Flask, jsonify, request
from config import *
import pandas
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import tweepy
import json

app = Flask(__name__)
app.config.from_object('config')

global tid
global uid
global api
global gnb
global gnb_u


@app.route("/set_uid", methods=['POST'])
def set_uid():
    global uid
    data = json.loads(request.data)
    uid = data['uid']

    response = jsonify(success=True)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/get_user", methods=['GET'])
def get_user():
    global uid
    global api
    global gnb_u

    users = api.lookup_users(user_ids=[uid])

    userProfiles = []

    for user in users:
        newUserProfile = {
            "follower_count": user.followers_count,
            "friends_count": user.friends_count,
            "statuses_count": user.statuses_count,
            "favourites_count": user.favourites_count
        }
        userProfiles.append(newUserProfile)



    userFrame = pandas.DataFrame(userProfiles)
    y_pred = gnb_u.predict(userFrame)

    response = jsonify({'is_bot': y_pred.item(0)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/set_tid", methods=['POST'])
def set_tid():
    global tid
    data = json.loads(request.data)
    tid = data['tid']

    response = jsonify(success=True)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/get_tweet", methods=['GET'])
def get_tweet():
    global tid
    global api
    global gnb
    global uid

    status = api.get_status(tid)
    tweets = []
    newTweet = {
        'retweet_count': status.retweet_count,
        'favourite_count': status.favorite_count,
        'word_count': len(status.text.split()),
        'sentiment': TextBlob(status.text).sentiment.polarity
    }
    uid = status.user.id_str
    tweets.append(newTweet)
    tweetFrame = pandas.DataFrame(tweets)
    y_pred = gnb.predict(tweetFrame)

    response = jsonify({'is_bot': y_pred.item(0), 'text': status.text, 'name': status.user.name})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/test", methods=['GET'])
def test():


    response = jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/start", methods=['POST'])
def start():
    global api
    global gnb
    global gnb_u

    consumer_key = '5pUecX1AHZlvqgnqgkJyhNsC4'
    consumer_secret = 'JgQsHCmRMDJjQppKUIw9HaEcw7VXK7GCd0a0mWiTZyoTVBZNvP'
    access_token = '392017792-Iovtp16Htzv1elBdflpsgggR3dC21PVMEpkIkHgq'
    access_token_secret = 'BZu75FAvopZreIyfkh9eFEfxBY2VMOJTCXYL3o7LBuCAP'

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    gnb = RandomForestClassifier()

    # IMPORT FAKE TWEETS AND DELETE UNEEDED COLUMNS
    names = ["id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","retweeted_status_id","geo","place","contributors","retweet_count","reply_count","favorite_count","favorited","retweeted","possibly_sensitive","num_hashtags","num_urls","num_mentions","created_at","timestamp","crawled_at","updated"]
    dataset = pandas.read_csv('tweets.txt', names=names)

    del dataset['crawled_at']
    del dataset['updated']
    del dataset['timestamp']
    del dataset['created_at']
    del dataset['num_mentions']
    del dataset['num_urls']
    del dataset['num_hashtags']
    del dataset['possibly_sensitive']
    del dataset['retweeted']
    del dataset['favorited']
    del dataset['geo']
    del dataset['place']
    del dataset['contributors']
    del dataset['truncated']
    del dataset['source']

    # ADD NEW COLUMN SAYING THEY ARE A BOT
    dataset['is_bot'] = 1

    # IMPORT REAL TWEETS AND DELETE UNEEDED COLUMNS

    names = ["id","text","source","user_id","truncated","in_reply_to_status_id","in_reply_to_user_id","in_reply_to_screen_name","retweeted_status_id","geo","place","contributors","retweet_count","reply_count","favorite_count","favorited","retweeted","possibly_sensitive","num_hashtags","num_urls","num_mentions","created_at","timestamp","crawled_at","updated"]
    real_dataset = pandas.read_csv('realtweets.txt', names=names)

    del real_dataset['crawled_at']
    del real_dataset['updated']
    del real_dataset['timestamp']
    del real_dataset['created_at']
    del real_dataset['num_mentions']
    del real_dataset['num_urls']
    del real_dataset['num_hashtags']
    del real_dataset['possibly_sensitive']
    del real_dataset['retweeted']
    del real_dataset['favorited']
    del real_dataset['geo']
    del real_dataset['place']
    del real_dataset['contributors']
    del real_dataset['truncated']
    del real_dataset['source']

    # ADD NEW COLUMN SAYING THEY ARE NOT A BOT
    real_dataset['is_bot'] = 0

    # Merge datasets
    frames = [dataset,real_dataset]
    full = pandas.concat(frames)

    # Randomize order
    final = full.sample(frac=1)

    # Get word count
    final['word_count'] = 0
    final['sentiment'] = 0.0
    for index, row in final.iterrows():
        final.at[index,'word_count'] = len(row['text'].split())
        final.at[index,'sentiment'] = TextBlob(row['text']).sentiment.polarity


    # Set columns we want to base prediction off
    feature_cols = ['retweet_count','favorite_count', 'word_count', 'sentiment']
    # feature_cols = ['retweet_count','reply_count','favorite_count', 'word_count','sentiment']


    # Convert values in featured columns to floats
    for col in feature_cols:
        final[col] = pandas.to_numeric(final[col], errors='coerce')
        final = final.dropna(subset=[col])
        final[col] = final[col].astype(float)

    # Split dataset and assign x and y values
    train, test = train_test_split(final, test_size=0.2)
    X = train.loc[:, feature_cols]
    y = train.is_bot

    gnb = SVC()
    gnb.fit(final[feature_cols].values, final["is_bot"])

    names = ["id","name","screen_name","statuses_count","followers_count","friends_count","favourites_count","listed_count","url","lang","time_zone","location","default_profile","default_profile_image","geo_enabled","profile_image_url","profile_banner_url","profile_use_background_image","profile_background_image_url_https","profile_text_color","profile_image_url_https","profile_sidebar_border_color","profile_background_tile","profile_sidebar_fill_color","profile_background_image_url","profile_background_color","profile_link_color","utc_offset","is_translator","follow_request_sent","protected","verified","notifications","description","contributors_enabled","following","created_at","timestamp","crawled_at","updated","test_set_1","test_set_2"]
    real_dataset = pandas.read_csv('realusers.txt', names=names)
    fake_dataset = pandas.read_csv('fakeusers.txt', names=names)

    real_dataset['is_bot'] = 0
    fake_dataset['is_bot'] = 1

    # Merge datasets
    frames = [fake_dataset,real_dataset]
    full = pandas.concat(frames)

    # Randomize order
    final = full.sample(frac=1)

    # Set columns we want to base prediction off
    feature_cols = ['statuses_count','followers_count','friends_count', 'favourites_count']

    # Convert values in featured columns to floats
    for col in feature_cols:
        final[col] = pandas.to_numeric(final[col], errors='coerce')
        final = final.dropna(subset=[col])
        final[col] = final[col].astype(float)

    # Split dataset and assign x and y values
    train, test = train_test_split(final, test_size=0.2)
    X = train.loc[:, feature_cols]
    y = train.is_bot

    gnb_u = DecisionTreeClassifier()
    gnb_u.fit(final[feature_cols].values, final["is_bot"])

    response = jsonify(success=True)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    app.run(debug=True)
