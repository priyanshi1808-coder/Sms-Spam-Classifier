import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



def transform_text(text):
    text = text.lower()
    text = text.split()  # simple split based on spaces

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in ENGLISH_STOP_WORDS and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # preprocess
    transformed_sms = transform_text(input_sms)
    # vectorize
    vector_input = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_input)[0]
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")