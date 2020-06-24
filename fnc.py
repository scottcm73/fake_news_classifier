import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
import csv
from stopwds import stopwords
import re
import string
import joblib
import pickle
def get_data():
    train_data=[[]]
    test_data=[]
    data_path1=os.path.join("resources", "train.csv" )
    data_path2=os.path.join("resources", "train2.csv" )
    data_path3=os.path.join("resources", "test.csv")
    train_df1=pd.read_csv(data_path1, index_col=False)

    
        
        
    
       

     
        
    y_train=train_df1['label']

            
    return train_df1
def clean_news_list(news_list):
    news_list_copy=news_list
    print('news_list_copy.shape')
    print(news_list_copy.shape)
    
    for row in news_list_copy:
    #Takes out most punctuation

        # Strip quotes
        np.array([re.sub(r'\"', '' ,a) for a in row[1]])
        np.array([re.sub(r'\"', '' ,a) for a in row[3]])

        np.array([re.sub(r'\W+', ' ' ,a) for a in row[1]])
        np.array([re.sub(r'\W+', ' ' ,a) for a in row[3]])
        # Strip digits
        np.array([re.sub(r'\d+', '' ,a) for a in row[1]])
        np.array([re.sub(r'\d+', '' ,a) for a in row[3]])
        # Strip hyphens
        np.array([re.sub(r'/-/g', ' ' ,a) for a in row[1]])
        np.array([re.sub(r'/-/g', ' ' ,a) for a in row[3]])
        row[1]=str(row[1])
        row[3]=str(row[3])
        ' '.join([word for word in row[1].split() if word not in (stopwords)])
        ' '.join([word for word in row[3].split() if word not in (stopwords)])
        return news_list_copy


def clean_news_df(df):
        
    news_df=df
    news_df = news_df.replace(np.nan, None)
    # news_df.drop(news_df[(news_df['label'] != 1) & (news_df['label']!=0)].index, inplace = True)
    # news_df.drop(news_df[(news_df['text'].str.isalnum()==False) & (news_df['title'].str.isalnum()==False)].index, inplace=True)
    # news_df.drop(news_df[news_df['label'].astype(str).str.isalpha()==True].index, inplace=True)
    
    news_df['title']=news_df['title'].str.lower()
    news_df['author']=news_df['author'].str.lower()
    news_df['text']=news_df['text'].str.lower()


    news_df['title'].replace(r'[^\w\s]|\d+|', '', regex=True, inplace=True)
    
    news_df['text'].replace(r'[^\w\s]|\d+', '', regex=True, inplace=True)


    news_df['title'].replace(r'\s+|\\n', ' ', regex=True, inplace=True) 
    news_df['text'].replace(r'\s+|\\n', ' ', regex=True, inplace=True) 
    
    
   
    news_df['title'].apply(lambda x: ([word for word in str(x).split() if word not in (stopwords)]))
    news_df['text'].apply(lambda x: ([word for word in str(x).split() if word not in (stopwords)]))
        
        
    
    return news_df




def train_model(X_train):
    news_df=X_train
    hash_text=HashingVectorizer(ngram_range=(3, 7), analyzer="char", alternate_sign=False)
    hash_title=HashingVectorizer(ngram_range=(3, 7), analyzer="char", alternate_sign=False)
    hash_author=HashingVectorizer(ngram_range=(3, 7), analyzer="word", alternate_sign=False)

   
    X_text=news_df['text']


    hash_text.fit(X_text)
    X = hash_text.fit_transform(X_text.values.astype('U'))
    X_title_text=news_df['title']

    X2 = hash_title.fit_transform(X_title_text.values.astype('U'))
    

    X_author=news_df['author']

    X3= hash_author.fit_transform(X_author.values.astype('U'))
    print('vectorized')
   
    pickle_path1=os.path.join("resources", "X_text_matrix.pkl")
    pickle_path2=os.path.join("resources", "X_title_matrix.pkl")
    pickle_path3=os.path.join("resources", "X_author_matrix.pkl")
    with open(pickle_path1, "wb") as output_file:
        pickle.dump(X, output_file)

    with open(pickle_path2, "wb") as output_file2:
        pickle.dump(X2, output_file2)
 
    with open(pickle_path3, "wb") as output_file3:
        pickle.dump(X3, output_file3)

    return
class model_test():
    def __init__(self):
        return

    def vectorize(self, X_text):
        news_df=X_text
    
        hash_text=HashingVectorizer(ngram_range=(3, 7), analyzer="char", alternate_sign=False)
        hash_title=HashingVectorizer(ngram_range=(3, 7), analyzer="char", alternate_sign=False)
        hash_author=HashingVectorizer(ngram_range=(3, 7), analyzer="char", alternate_sign=False)

   
        X_text=news_df['text']


        hash_text.fit(X_text)
        text_vector = hash_text.fit_transform(X_text.values.astype('U'))
        
        self.text_vector=text_vector
        X_title_text=news_df['title']

        print(text_vector[:1])

        title_vector= hash_title.fit_transform(X_title_text.values.astype('U'))
        self.title_vector=title_vector

        X_author=news_df['author']

        author_vector = hash_author.fit_transform(X_author.values.astype('U'))
        self.author_vector=author_vector
   
   
        return  author_vector

    def test_model(self, X_test):
        pickle_path1=os.path.join("resources", "X_text_matrix.pkl")
        pickle_path2=os.path.join("resources", "X_title_matrix.pkl")
        pickle_path3=os.path.join("resources", "X_author_matrix.pkl")
        with open(pickle_path1, "rb") as output_file:
            X1=pickle.load(output_file)

        with open(pickle_path2, "rb") as output_file2:
            X2=pickle.load(output_file2)
 
        with open(pickle_path3, "rb") as output_file3:
            X3=pickle.load(output_file3)
        print(X3[:5])

        clf1 = ComplementNB().fit(X3, y_train)
        clf2 = ComplementNB().fit(X2, y_train)
        clf3 = ComplementNB().fit(X1, y_train)
        print(clf3)

        X4 = self.vectorize(X_test)
        test_predict=clf1.predict(X4)
        author_predict = np.asarray(test_predict, dtype=np.float64, order='C')

        X5 = self.title_vector
        test_predict2=clf2.predict(X5)
        title_predict = np.asarray(test_predict2, dtype=np.float64, order='C')
        self.title_predict=title_predict
        X6 = self.text_vector

        text_predict=clf3.predict(X6)
        #text_predict = np.asarray(test_predict3, dtype=np.float64, order='C')
        

     
        

  
        self.author_predict=author_predict
        self.title_predict=title_predict
        self.text_predict=text_predict
        return 

train_df=get_data()
# print('train_df')
# print(train_df.head())

# print(train_df.shape)


clean_news=clean_news_df(train_df)
# print('clean_news')
# print(clean_news)

X=clean_news[['title', 'author', 'text']]
y=clean_news[['label']].values.ravel()
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=5)
# print('X_train.head -- cleaned')
# print(X_train.head())
# train_model(X_train)

# clf=train_model(X_train)



the_test=model_test()
y_pred1=the_test.test_model(X_test)

y_pred1=the_test.author_predict


y_pred2=the_test.title_predict

y_pred3=the_test.text_predict

print('y_predictions')
print(y_pred2)
print(y_pred1)
print(y_pred3)
