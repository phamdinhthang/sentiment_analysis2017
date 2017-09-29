import text_processor as tp
import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize


##----------Processor----------

def sentiments_classify(text,positive_words,negative_words):
    positive_score = tp.jaccard_word_index(text,' '.join(positive_words))
    negative_score = tp.jaccard_word_index(text,' '.join(negative_words))
    ##print('Positive score = ' + str(positive_score) + ', Negative score = ' + str(negative_score))
    if positive_score > negative_score:
        return 'positive'
    elif positive_score < negative_score:
        return 'negative'
    else:
        return 'neutral'
    
def batch_sentiments_classify(corpus,positive_words,negative_words):
    res = []
    for text in corpus:
        res.append(sentiments_classify(text,positive_words,negative_words))
    return res

def validate_sentiment_analysis_result(label,predicted,neutral_as_true=True):
    if len(predicted) == len(label):
        correct_cnt = 0
        for index,element in enumerate(predicted):
            if element == 'neutral':
                if neutral_as_true == True:
                    correct_cnt += 1
            elif element == 'positive':
                if label[index] == 1:
                    correct_cnt += 1
            elif element == 'negative':
                if label[index] == 0:
                    correct_cnt += 1
    else:
        print('Error. Pls check result and label length')
    return correct_cnt/len(predicted)

def get_corpus_and_sentiments(path,text_column_name,sentiment_column_name):
    df = pd.read_csv(path,sep=',',error_bad_lines=False,quotechar='"',encoding = "ISO-8859-1")
    corpus = df[text_column_name].values
    corpus = tp.pre_processing(corpus)
    sentiment = df[sentiment_column_name].values
    print('---Preview corpus: ' + str(corpus[1:10]))
    return (corpus,sentiment,df)

##----------Experiments----------

#Environment setting
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#Import sentiments lexicon
positive_path = dname + '/sample_data/positive_words.txt'
negative_path = dname + '/sample_data/negative_words.txt'
with open(positive_path,encoding='utf-8') as file:
    positive_words = file.readlines()
with open(negative_path,encoding='utf-8') as file:
    negative_words = file.readlines()
positive_words = [line.strip() for line in positive_words] 
negative_words = [line.strip() for line in negative_words]
print('---Sample positive lexicon: ' + str(positive_words[0:10]))
print('---Sample negative lexicon: ' + str(negative_words[0:10]))


#Test with multiple dataset
dataset_names = ['tweets_10000.csv','amazon.csv','imdb.csv','yelp.csv']
for name in dataset_names: 
    path = dname + '/sample_data/' + name
    data = get_corpus_and_sentiments(path,'text','sentiment')

    corpus = data[0]
    sentiment = data[1]

    predicted = batch_sentiments_classify(corpus,positive_words,negative_words)
    accuracy = validate_sentiment_analysis_result(sentiment,predicted,True)
    print('Accuracy for ' + name + ' dataset = ' + str(accuracy))

##Test with sample text for the policy
text1 = 'I do not want to change my living place'
text2 = 'The government policy is great. People would love to switch to new flats instead of old ones'
text3 = 'What the hell is this policy ? Moving citizen without asking them ?'
text4 = 'I like the policy. I will move'
text5 = 'There are 4 generations in my family and some of them will hate the new policy'
texts = [text1,text2,text3,text4,text5]
for text in texts:
    print(text + ' --> ' + sentiments_classify(text,positive_words,negative_words))
