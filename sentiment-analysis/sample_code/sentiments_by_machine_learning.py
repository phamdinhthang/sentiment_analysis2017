import text_processor as tp
import os
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize


#Environment setting
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#Classifier train and test
def naive_bayes_sentiments_classifier_train(df,printAccuracy=True):
    df_positive = df[df['sentiment']==1]
    df_negative = df[df['sentiment']==0]
    
    corpus = df['text'].values
    corpus_positive = df_positive['text'].values
    corpus_negative = df_negative['text'].values
    corpus = tp.pre_processing(corpus)
    corpus_positive = tp.pre_processing(corpus_positive)
    corpus_negative = tp.pre_processing(corpus_negative)

    positive_features = []
    negative_features = []
    for text in corpus_positive:
        positive_features.append((tp.create_word_features(tp.bag_of_words(text)), 'positive'))
    for text in corpus_negative:
        negative_features.append((tp.create_word_features(tp.bag_of_words(text)), 'negative'))

    #Create training set and testing set and test for error
    positive_split_index = round(0.7*len(positive_features))
    negative_split_index = round(0.7*len(negative_features))

    train_set = negative_features[:negative_split_index] + positive_features[:positive_split_index]
    test_set =  negative_features[negative_split_index:] + positive_features[positive_split_index:]

    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.util.accuracy(classifier, test_set)
    if printAccuracy == True:
        print("Naive bayes classifier accuracy: " + str(accuracy))
    return classifier
 
def naive_bayes_sentiments_classifier_test(text,classifier):
    bows = tp.bag_of_words(text)
    words = tp.create_word_features(bows)
    return classifier.classify(words)

def knn_sentiments_classifier_train(df,printAccuracy=True):
    corpus = df['text'].values
    labels = df['sentiment'].values
    
    tokenize = lambda doc: doc.lower().split(' ')
    tf_idf = TfidfVectorizer(max_features=2000,norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
    dtm = tf_idf.fit_transform(corpus)
    feature_names = tf_idf.get_feature_names()

    features_train, features_test, labels_train, labels_test = train_test_split(dtm, labels, test_size = 0.3, random_state=42, stratify=labels)
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(features_train,labels_train)
    if printAccuracy == True:
        print('KNN classifier accuracy = ',knn.score(features_test,labels_test))
    return knn

def knn_sentiments_classifier_test(text,classifier,df):
    text_to_vect = tp.get_word_vector(text,tp.get_tf_idf_terms(df))
    predicted = classifier.predict(text_to_vect)
    if predicted == 1:
        return 'positive'
    if predicted == 0:
        return 'negative'

#Test with multiple dataset
dataset_names = ['tweets_10000.csv','amazon.csv','imdb.csv','yelp.csv']
for name in dataset_names:
    print('--------------dataset: ' + name + ' -------------------')
    path = dname + '/sample_data/' + name
    df = pd.read_csv(path,sep=',',error_bad_lines=False,quotechar='"',encoding = "ISO-8859-1")
    naive_bayes_sentiments_classifier_train(df)
    knn_sentiments_classifier_train(df)

#Test with sample text for the policy
text1 = 'I do not want to change my living place'
text2 = 'The government policy is great. People would love to switch to new flats instead of old ones'
text3 = 'What the hell is this policy ? Moving citizen without asking them ?'
text4 = 'I like the policy. I will move'
text5 = 'There are 4 generations in my family and some of them will hate the new policy'
texts = [text1,text2,text3,text4,text5]

#Use the amazon dataset to train
path = dname + '/sample_data/amazon.csv'
df = pd.read_csv(path,sep=',',error_bad_lines=False,quotechar='"',encoding = "ISO-8859-1")
nb = naive_bayes_sentiments_classifier_train(df,False)
knn = knn_sentiments_classifier_train(df,False)

print('--------------Naive Bayes sentiment classifier-------------------')
for text in  texts:
    print(text + ' --> ' + naive_bayes_sentiments_classifier_test(text,nb))
print('--------------KNN sentiment classifier-------------------')
for text in  texts:
    print(text + ' --> ' + knn_sentiments_classifier_test(text,knn,df))

