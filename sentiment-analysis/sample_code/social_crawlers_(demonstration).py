import facebook
import urllib3 
import requests
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize


##---------Demonstration-------------

def getTopicsFeatures(text):
    ##some features extraction
    subject = 'New Site Identified for Selective En bloc Redevelopment Scheme'
    keywords = ['singapore','government','blocks','513 to 520','west coast road','SERS','99-year','replacement flats','commercial properties']
    location = {'lat':1.300395,'long':103.848610,'radius':10}
    time_span = {'start':'03/08/2016','end':'03/09/2016'}
    search_dic = {'subject':subject,'keywords':keywords,'location':location,'time_span':time_span}
    return search_dic

def crawlFromFacebook(searchItem):
    #Facebook graph api connection
    facebook_token= 'some sample token'
    graph = facebook.GraphAPI(access_token=token, version = 2.7)
    facebook_entity = ['page','group','event','user']
    facebook_text_corpus = []

    #Facebook post crawlers
    for entity in facebook_entity:
        for keyword in searchItem['keywords']:
            data = graph.request('/search?q='+keyword+'&type='+entity+'&limit=100')
            dataList = pages['data']
            for dt in dataList:
                dt_id = dt['id']
                posts = graph.request('/'+dt_id+'/search?q='+keyword+'&type=post&limit=100')
                postsList = posts['data']
                for post in postsList:
                    post_id = post['id']
                    post_data = graph.get_object(id=event_id,fields='description,message,message_tags,picture,place,properties,source,status_type,story')
                    post_message = post_data['message']
                    post_contents = requests.get('https://graph.facebook.com/v2.7/'+post_id+'/message?access_token='+token+'&limit=100')
                    post_json = post_contents.json()
                    facebook_text_corpus.append(post_json)
    return facebook_text_corpus

def crawlFromTwitter(searchItem):
    #Twitter API connection
    consumer_key = 'some sample key'
    consumer_secret = 'some sample secret'
    access_token = 'some sample token'
    access_token_secret = 'some sample token secret'
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    #Tweets crawler
    tweets_text_corpus = []
    for keyword in searchItem['keywords']:
        fetched_tweets = api.search(q = keyword, count = 100)
        for tweet in fetched_tweets:
            tweets_text_corpus.append(tweet.text)
    return tweets_text_corpus

