# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:33:04 2023

@author: Admin
"""
# importing pandas to read the csv file
import pandas as pd

#here we read the file and sep is for seperation in the data set we have 1 tab space for the
#lable and the message and we name them as respectively.
messages = pd.read_csv('C:/Users/Admin/Desktop/NLP/messagesamp/SMSSpamCollection', sep="\t",
                       names =["lable","message"])

# Data Cleaning and Preparation
import re # regularization
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps = PorterStemmer() # cerating the object
corpus= []
for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review =[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
#before giving the max_features it gave the 6296 columns so i gave 5000

#Seperating the independent features
X = cv.fit_transform(corpus).toarray()
#sperating the dependent features and converting the ham and spam dummey variables
y=pd.get_dummies(messages['lable'])
y=y.iloc[:,1].values

#Train-test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training model using Naviebayes classiifer
from sklearn.naive_bayes import MultinomialNB
spam_detect_model= MultinomialNB().fit(X_train,y_train)
#prediction
y_pred = spam_detect_model.predict(X_test)

#to comparision
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)

#Accuracy score of the prediction
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) 