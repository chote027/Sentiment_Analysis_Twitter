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

app = Flask(__name__)

plt.style.use('fivethirtyeight')

login = pd.read_csv('key.csv')
consumer_key = login['key'][0]
consumer_secret = login['secret'][0]
access_token = login['token'][0]
access_token_secret = login['token_secret'][0]

#auth obj
authenticate = tweepy.OAuthHandler(consumer_key, consumer_secret)

#set access token
authenticate.set_access_token(access_token, access_token_secret)

#create api obj
api = tweepy.API(authenticate, wait_on_rate_limit = True)

# extract 100 recent tweets from twitter user timelime
def getTweetFromUser(query):
    posts = api.user_timeline(screen_name=query, count=100, result_type='recent', lang='th', tweet_mode='extended')
    df = pd.DataFrame({'Tweets': [tweet.full_text for tweet in posts],
                   'StatusID': [tweet.id_str for tweet in posts],
                   'UserID': [tweet.user.screen_name for tweet in posts],
                   'Createdat': [tweet.created_at for tweet in posts]})
    df['Tweets'] = df['Tweets'].apply(cleanText)
    return df

#extract 100 recent tweets from twitter hashtag
def getTweet(query):
    # create dataframe
    posts = api.search(q=query, count=100, result_type='recent', lang='th', tweet_mode='extended')
    df = pd.DataFrame({'Tweets': [tweet.full_text for tweet in posts],
                   'StatusID': [tweet.id_str for tweet in posts],
                   'UserID': [tweet.user.screen_name for tweet in posts],
                   'Createdat': [tweet.created_at for tweet in posts]})
    df['Tweets'] = df['Tweets'].apply(cleanText)
    return df

#create function to clean the tweets
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9_]+:', '', text) # remove #mentions
    text = re.sub(r'@[A-Za-z0-9_]+', '', text) # remove #mentions
    text = re.sub(r'#','',text) # remove hashtag
    text = re.sub(r'RT[\s]+','', text) # remove RT
    text = re.sub(r'https?:\/\/\S+','', text) # remove hyper link
    return text

model = joblib.load('frozen_model/sent_model.pkl')
scaler_fit = joblib.load('frozen_model/scaler_model.pkl')
tfidf_fit = joblib.load('frozen_model/tfidf_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictbyhashtag', methods=['GET','POST'])
def predictbyhashtag():
    if request.method=='POST':
        # text + userid + date&time
        hashtag = request.form['hashtag']
        posts = getTweet(hashtag)
        if posts.shape[0] != 0 :
            posts["processed"] = posts.Tweets.map(lambda x: "|".join(process_thai(x)))
            posts["wc"] = posts.processed.map(lambda x: len(x.split("|")))
            posts["uwc"] = posts.processed.map(lambda x: len(set(x.split("|"))))

            tf_input = tfidf_fit.transform(posts["Tweets"])

            num_input = scaler_fit.transform(posts[["wc","uwc"]].astype(float))

            t_input =  np.concatenate([num_input,tf_input.toarray()],axis=1)

            output_pd = pd.DataFrame(model.predict_proba(t_input))
            output_pd.columns = model.classes_
            output_pd["Predict"] = model.predict(t_input)
            output_pd["Tweets"] = posts.Tweets
            output_pd["Processed"] = posts.processed
            output_pd["wc"] = posts.wc
            output_pd["uwc"] = posts.uwc
            output_pd["StatusID"] = posts["StatusID"]
            output_pd["UserID"] = posts["UserID"]
            output_pd["Createdat"] = posts["Createdat"]
            output_pd['Predict'] = output_pd['Predict'].replace({'neg':'Negative', 'pos':'Positive','neu':'Neutral'})
            output_pd = output_pd.rename(columns={"neg":"Negative","neu":"Neutral","pos":"Positive"})
            result = output_pd.values.tolist()
            return render_template('output.html' , output_result=result , length = len(result))
        else : 
            return render_template('output.html' , length = 0)
    return render_template('predictByHashtag.html')

@app.route('/predictbyuserID', methods=['GET','POST'])
def predictbyid():
    if request.method=='POST':
        # text + userid + date&time
        user = request.form['userID']
        posts = getTweetFromUser(user)
        if posts.shape[0] != 0 :
            posts["processed"] = posts.Tweets.map(lambda x: "|".join(process_thai(x)))
            posts["wc"] = posts.processed.map(lambda x: len(x.split("|")))
            posts["uwc"] = posts.processed.map(lambda x: len(set(x.split("|"))))

            tf_input = tfidf_fit.transform(posts["Tweets"])

            num_input = scaler_fit.transform(posts[["wc","uwc"]].astype(float))

            t_input =  np.concatenate([num_input,tf_input.toarray()],axis=1)

            output_pd = pd.DataFrame(model.predict_proba(t_input))
            output_pd.columns = model.classes_
            output_pd["Predict"] = model.predict(t_input)
            output_pd["Tweets"] = posts.Tweets
            output_pd["Processed"] = posts.processed
            output_pd["wc"] = posts.wc
            output_pd["uwc"] = posts.uwc
            output_pd["StatusID"] = posts["StatusID"]
            output_pd["UserID"] = posts["UserID"]
            output_pd["Createdat"] = posts["Createdat"]
            output_pd['Predict'] = output_pd['Predict'].replace({'neg':'Negative', 'pos':'Positive','neu':'Neutral'})
            output_pd = output_pd.rename(columns={"neg":"Negative","neu":"Neutral","pos":"Positive"})
            result = output_pd.values.tolist()
            return render_template('output.html' , output_result=result , length = len(result))
        else : 
            return render_template('output.html' , length = 0)
    return render_template('predictById.html')

    
@app.route('/predictbysentence', methods=['GET','POST'])
def predictbysentence():
    result = ""
    if request.method=='POST':
        text = request.form['texts']
        texts = cleanText(text)
        test_input = pd.DataFrame({"texts":[texts]})
        if test_input.shape[0] != 0 :
            test_input["processed"] = test_input.texts.map(lambda x: "|".join(process_thai(x)))
            test_input["wc"] = test_input.processed.map(lambda x: len(x.split("|")))
            test_input["uwc"] = test_input.processed.map(lambda x: len(set(x.split("|"))))

            tf_input = tfidf_fit.transform(test_input["texts"])

            num_input = scaler_fit.transform(test_input[["wc","uwc"]].astype(float))

            t_input =  np.concatenate([num_input,tf_input.toarray()],axis=1)

            output_pd = pd.DataFrame(model.predict_proba(t_input))
            output_pd.columns = model.classes_
            output = model.predict(t_input)
            if output == "neg":
                result = "Negative"
            elif output == "pos":
                result = "Positive"
            elif output == "neu":
                result = "Neutral"
            # output_pd["Prediction"] = model.predict(t_input)
            # output_pd["Texts"] = test_input.texts
            # output_pd["Processed"] = test_input.processed
            # output_pd["Wc"] = test_input.wc
            # output_pd["Uwc"] = test_input.uwc
            # output_pd['Predict'] = output_pd['Predict'].replace({'neg':'Negative', 'pos':'Positive','neu':'Neutral'})
            # output_pd = output_pd.rename(columns={"neg":"Negative","neu":"Neutral","pos":"Positive"})
            # result = output_pd
            return render_template('outputBySentence.html' , result=result)
        else : 
            return render_template('outputBySentence.html' , length = 0)
    return render_template('predictBySentence.html')


if __name__ == '__main__':
    app.run(debug=True)

