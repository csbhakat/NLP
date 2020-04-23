# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:31:38 2020

@author: Chandra Shekhar Bhakat, amazon interview quetion 1- Urdu SMS sentiment analysis
"""
import pandas as pd

df=pd.read_csv('C:/Users/chbhakat/Desktop/DS/Projects/SentimentAnalysis/Roman Urdu DataSet.csv')
df.columns=['message','sentiment','notuseful']

df.drop('notuseful',axis=1,inplace=True)
df[df['message'].isnull()]
df['message'].fillna('jee ye to',inplace=True)
df.loc[df['sentiment']=='Neative','sentiment']='Negative'

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = review.split()
    
    #review = [ps.stem(word) for word in review if not word in stopwords.words('urdu')]
    review = ' '.join(review)
    corpus.append(review)
    

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

#y=pd.get_dummies(df['sentiment'])

d={'Positive':1,'Negative':0,'Neutral':42}
y=df['sentiment'].map(d)
#y=pd.factorize( ['Negative', 'Neutral', 'Positive'] )[0]
#y=y.iloc[:,2].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

#T--------------------------------Using bag of words ---------------------------------------
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
score=accuracy_score(y_test,y_pred)
print(score)
report=classification_report(y_test,y_pred)
print(report)


from sklearn.ensemble import RandomForestClassifier


# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(X_train,y_train)
predict=randomclassifier.predict(X_test)

#To calculate accuracy of bag of words
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix_RF=confusion_matrix(y_test,predict)
print(matrix_RF)
score_RF=accuracy_score(y_test,predict)
print(score_RF)
report_RF=classification_report(y_test,predict)
print(report_RF)

#----------------------------------Using TF-IDF --------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X_TF_IDF = cv.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_TF_IDF, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred_TF_IDF=spam_detect_model.predict(X_test)

#To calculate accuracy of bag of words
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix_TF_IDF=confusion_matrix(y_test,y_pred_TF_IDF)
print(matrix_TF_IDF)
score_TF_IDF=accuracy_score(y_test,y_pred_TF_IDF)
print(score_TF_IDF)
report_TF_IDF=classification_report(y_test,y_pred_TF_IDF)
print(report_TF_IDF)


# --------------------------------Using SVM--------------------------------------------------- 
from sklearn.svm import SVC
svm=SVC(random_state=1)
svm.fit(X_train,y_train)

y_pred_SVM=spam_detect_model.predict(X_test)

#To calculate accuracy of bag of words
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix_SVM=confusion_matrix(y_test,y_pred_SVM)
print(matrix_SVM)
score_SVM=accuracy_score(y_test,y_pred_SVM)
print(score_SVM)
report_SVM=classification_report(y_test,y_pred_SVM)
print(report_SVM)


# --------------------------------Using Word to vec-----------------------------------------
# Training the Word2Vec model
from gensim.models import Word2Vec
X_WV = Word2Vec(corpus, min_count=2)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_WV, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred_WV=spam_detect_model.predict(X_test)

#To calculate accuracy of bag of words
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
matrix_WV=confusion_matrix(y_test,y_pred_WV)
print(matrix_WV)
score_WV=accuracy_score(y_test,y_pred_WV)
print(score_WV)
report_WV=classification_report(y_test,y_pred_WV)
print(report_WV)







