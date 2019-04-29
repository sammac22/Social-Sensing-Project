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


#JUST LIKE OTHER FILE, BELOW THIS IS THE IMPORTANT FUNCTION THAT NEEDS TO BE CALLED, ABOVE THIS NEEDS TO BE RUN ON STARTUP
# PLACE THE USER ID IN THE id VARIABLE AND RETURN WHETHER OR NOT IT IS A BOT

id = '10788822'

users = api.lookup_users(user_ids=[id])

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
y_pred = gnb.predict(userFrame)
print(y_pred)

