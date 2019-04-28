import csv
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from collections import defaultdict
from textblob import TextBlob
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import tweepy

consumer_key = '5pUecX1AHZlvqgnqgkJyhNsC4'
consumer_secret = 'JgQsHCmRMDJjQppKUIw9HaEcw7VXK7GCd0a0mWiTZyoTVBZNvP'
access_token = '392017792-Iovtp16Htzv1elBdflpsgggR3dC21PVMEpkIkHgq'
access_token_secret = 'BZu75FAvopZreIyfkh9eFEfxBY2VMOJTCXYL3o7LBuCAP'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


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


# Function for calculating accuracy scores of model

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    print("True positives: " + str(TP))
    print("False positives: " + str(FP))
    print("True negatives: " + str(TN))
    print("False negatives: " + str(FN))


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

# Create and train model
# models = [SVC,DecisionTreeClassifier,LogisticRegression,LinearDiscriminantAnalysis,KNeighborsClassifier,RandomForestClassifier]
# for model in models:
#     print(str(model))
#     gnb = model()
#     gnb.fit(train[feature_cols].values, train["is_bot"])
#
#     # Predict with model and print results
#     y_pred = gnb.predict(test[feature_cols])
#     print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(test.shape[0],(test["is_bot"] != y_pred).sum(),100*(1-(test["is_bot"] != y_pred).sum()/test.shape[0])))
#
#     tn, fp, fn, tp = confusion_matrix(test['is_bot'], y_pred).ravel()
#     print("True positives: " + str(tp))
#     print("False positives: " + str(fp))
#     print("True negatives: " + str(tn))
#     print("False negatives: " + str(fn))

gnb = DecisionTreeClassifier()
gnb.fit(final[feature_cols].values, final["is_bot"])



#BELOW THIS IS THE IMPORTANT FUNCTION. WE NEED TO PUT THE ID OF THE TWEET WE WANT IN id_of_tweet VARIABLE
#THEN RETURN WHETHER OR NOT IT IS A BOT ALONG WITH THE ID OF THE ACCOUNT THAT TWEETED IT
#EVERYTHING ABOVE THIS NEEDS TO BE RUN ON STARTUP

id_of_tweet = ''
status = api.get_status(id_of_tweet)
tweets = []
newTweet = {
    'retweet_count': status.retweet_count,
    'favourite_count': status.favorite_count,
    'word_count': len(status.text.split()),
    'sentiment': TextBlob(status.text).sentiment.polarity
}
tweets.append(newTweet)
print(newTweet)
tweetFrame = pandas.DataFrame(tweets)
y_pred = gnb.predict(tweetFrame)
print(y_pred)



