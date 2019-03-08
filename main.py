import csv
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from collections import defaultdict

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

dataset['is_bot'] = 1

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

real_dataset['is_bot'] = 0

frames = [dataset,real_dataset]

full = pandas.concat(frames)

final = full.sample(frac=1)

d = defaultdict(preprocessing.LabelEncoder)

final.apply(preprocessing.LabelEncoder().fit_transform).astype(str)

train, test = train_test_split(final, test_size=0.2)

feature_cols = ['id', 'text', 'user_id', 'retweet_count','reply_count','favorite_count']

X = train.loc[:, feature_cols]
y = train.is_bot

gnb = GaussianNB()
gnb.fit(train[feature_cols].values, train["is_bot"])

y_pred = gnb.predict(test[feature_cols])

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(test.shape[0],(test["is_bot"] != y_pred).sum(),100*(1-(test["Survived"] != y_pred).sum()/test.shape[0])))

