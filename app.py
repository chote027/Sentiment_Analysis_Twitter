from types import MethodType
from flask import Flask, render_template, request, url_for, jsonify
import flask
import joblib
import pandas as pd
import numpy as np
from pythainlp.ulmfit import process_thai
import tweepy
import json
import re
import matplotlib.pyplot as plt
from flask_frozen import Freezer

app = Flask(__name__)
freezer = Freezer(app)

plt.style.use('fivethirtyeight')

# decalre key value from key.csv
login = pd.read_csv('key.csv')
consumer_key = login['key'][0]
consumer_secret = login['secret'][0]
access_token = login['token'][0]
access_token_secret = login['token_secret'][0]

# auth obj
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)

# set access token
authenticate.set_access_token(access_token, access_token_secret)

# create api obj
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# declare model
model = joblib.load('frozen_model/sent_model.pkl')
scaler_fit = joblib.load('frozen_model/scaler_model.pkl')
tfidf_fit = joblib.load('frozen_model/tfidf_model.pkl')

# create function to extract 100 recent tweets from twitter user timelime
def getTweetFromUser(query):
    # get tweets
    posts = api.user_timeline(screen_name=query, count=100, result_type='recent', lang='th', tweet_mode='extended')
    # create and put data into dataframe
    df = pd.DataFrame({'Tweets': [tweet.full_text for tweet in posts],
                   'StatusID': [tweet.id_str for tweet in posts],
                   'UserID': [tweet.user.screen_name for tweet in posts],
                   'Createdat': [tweet.created_at for tweet in posts]})
    # clean text
    df['Tweets'] = df['Tweets'].apply(cleanText)
    return df

# create function to extract 100 recent tweets from twitter hashtag
def getTweet(query):
    # get tweets
    posts = api.search(q=query, count=100, result_type='recent', lang='th', tweet_mode='extended')
    # create and put data into dataframe
    df = pd.DataFrame({'Tweets': [tweet.full_text for tweet in posts],
                   'StatusID': [tweet.id_str for tweet in posts],
                   'UserID': [tweet.user.screen_name for tweet in posts],
                   'Createdat': [tweet.created_at for tweet in posts]})
    # clean text
    df['Tweets'] = df['Tweets'].apply(cleanText)
    return df

# create function to clean the tweets
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9_]+:', '', text) # remove #mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # remove #mentions
    text = re.sub(r'#','',text) # remove hashtag
    text = re.sub(r'RT[\s]+','', text) # remove RT
    text = re.sub(r'https?:\/\/\S+','', text) # remove hyper link
    return text

# create function to predict and put output to dataframe
def predict(t_input, posts):
    # predict
    output_pd = pd.DataFrame(model.predict_proba(t_input))
    output_pd.columns = model.classes_
    output_pd["Predict"] = model.predict(t_input)
    # put others data into dataframe
    output_pd["Tweets"] = posts.Tweets
    output_pd["Processed"] = posts.processed
    output_pd["wc"] = posts.wc
    output_pd["uwc"] = posts.uwc
    output_pd["StatusID"] = posts["StatusID"]
    output_pd["UserID"] = posts["UserID"]
    output_pd["Createdat"] = posts["Createdat"]
    output_pd['Predict'] = output_pd['Predict'].replace({'neg':'Negative', 'pos':'Positive','neu':'Neutral'})
    output_pd = output_pd.rename(columns={"neg":"Negative","neu":"Neutral","pos":"Positive"})
    return output_pd.values.tolist()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictbyhashtag', methods=['GET','POST'])
def predictbyhashtag():
    if request.method=='POST':
        # get hashtag from text box
        hashtag = request.form['hashtag']
        # get tweets with hashtag
        posts = getTweet(hashtag)
        if posts.shape[0] != 0 :
            # words processing
            posts["processed"] = posts.Tweets.map(lambda x: "|".join(process_thai(x)))
            posts["wc"] = posts.processed.map(lambda x: len(x.split("|")))
            posts["uwc"] = posts.processed.map(lambda x: len(set(x.split("|"))))
            tf_input = tfidf_fit.transform(posts["Tweets"])
            num_input = scaler_fit.transform(posts[["wc","uwc"]].astype(float))
            t_input =  np.concatenate([num_input,tf_input.toarray()],axis=1)

            # predict and convert output to list
            result = predict(t_input, posts)
            return render_template('output.html' , output_result=result , length = len(result))
        else : 
            return render_template('output.html' , length = 0)
    return render_template('predictByHashtag.html')

@app.route('/predictbyuserID', methods=['GET','POST'])
def predictbyid():
    if request.method=='POST':
        # get user id from text box
        user = request.form['userID']
        # get tweets with user id
        posts = getTweetFromUser(user)
        if posts.shape[0] != 0 :
            # words processing
            posts["processed"] = posts.Tweets.map(lambda x: "|".join(process_thai(x)))
            posts["wc"] = posts.processed.map(lambda x: len(x.split("|")))
            posts["uwc"] = posts.processed.map(lambda x: len(set(x.split("|"))))
            tf_input = tfidf_fit.transform(posts["Tweets"])
            num_input = scaler_fit.transform(posts[["wc","uwc"]].astype(float))
            t_input =  np.concatenate([num_input,tf_input.toarray()],axis=1)

            # predict and convert output to list
            result = predict(t_input, posts)
            return render_template('output.html' , output_result=result , length = len(result))
        else : 
            return render_template('output.html' , length = 0)
    return render_template('predictById.html')

    
@app.route('/predictbysentence', methods=['GET','POST'])
def predictbysentence():
    result = ""
    if request.method=='POST':
        # get sentense from text box
        text = request.form['texts']
        # clean text
        texts = cleanText(text)
        # create and put data into dataframe
        posts = pd.DataFrame({"texts":[texts]})
        if posts.shape[0] != 0 :
            # words processing
            posts["processed"] = posts.texts.map(lambda x: "|".join(process_thai(x)))
            posts["wc"] = posts.processed.map(lambda x: len(x.split("|")))
            posts["uwc"] = posts.processed.map(lambda x: len(set(x.split("|"))))
            tf_input = tfidf_fit.transform(posts["texts"])
            num_input = scaler_fit.transform(posts[["wc","uwc"]].astype(float))
            t_input =  np.concatenate([num_input,tf_input.toarray()],axis=1)

            # predict
            output_pd = pd.DataFrame(model.predict_proba(t_input))
            output_pd.columns = model.classes_
            output = model.predict(t_input)
            # replace output with word
            if output == "neg":
                result = "Negative"
            elif output == "pos":
                result = "Positive"
            elif output == "neu":
                result = "Neutral"
            return render_template('outputBySentence.html' , result=result)
        else : 
            return render_template('outputBySentence.html' , length = 0)
    return render_template('predictBySentence.html')


if __name__ == '__main__':
    app.run(debug=True)

