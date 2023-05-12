import streamlit as st
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_model():
    review=pd.read_csv("reviews.csv")
    review=review.rename(columns={'text':'review'},inplace=False)
    x=review.review
    y=review.polarity

    x_train, x_test, y_train, y_test=train_test_split(x,y, train_size=0.6, random_state=1)

    vector=CountVectorizer(stop_words='english' ,lowercase=False)

    vector.fit(x_train)
    
    x_transformed = vector.transform(x_train)
    x_transformed.toarray()
    # for test data
    x_test_transformed = vector.transform(x_test)
    naivebayes = MultinomialNB()
    naivebayes.fit(x_transformed, y_train)
    

    s = pickle.dumps(naivebayes)
    model=pickle.loads(s)
    return vector and model

model,vector=train_model()
    

def predict(input):
    
    vec=vector.transform([input]).toarray()
    category=(str(list(model.predict(vec)[0])).replace("0","negative").replace("1","positive"))
    return category

st.header('Review')
input = st.text_area("Please enter the text")
if st.button('classify'):
    st.write(predict(input))

train_model()
predict()