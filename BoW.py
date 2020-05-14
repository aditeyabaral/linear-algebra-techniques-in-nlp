import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re, string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


contraction = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot", 
    "can't've": "cannot have", 
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
    "hadn't've": "had not have", 
    "hasn't": "has not",
    "haven't": "have not", 
    "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
    "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
    "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
    "I'll've": "I will have","I'm": "I am", "I've": "I have", 
    "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
    "i'll've": "i will have","i'm": "i am", "i've": "i have", 
    "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
    "it'll": "it will", "it'll've": "it will have","it's": "it is", 
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
    "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
    "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
    "she's": "she is", "should've": "should have", "shouldn't": "should not", 
    "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
    "this's": "this is",
    "that'd": "that would", "that'd've": "that would have","that's": "that is", 
       "there'd": "there would", "there'd've": "there would have","there's": "there is", 
       "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
       "they'll've": "they will have", "they're": "they are", "they've": "they have", 
       "to've": "to have", "wasn't": "was not", "we'd": "we would", 
       "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
       "we're": "we are", "we've": "we have", "weren't": "were not", 
       "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
       "what's": "what is", "what've": "what have", "when's": "when is", 
       "when've": "when have", "where'd": "where did", "where's": "where is", 
       "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
       "who's": "who is", "who've": "who have", "why's": "why is", 
       "why've": "why have", "will've": "will have", "won't": "will not", 
       "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
       "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
       "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
       "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
       "you'll've": "you will have", "you're": "you are", "you've": "you have"}

stopw = set(stopwords.words('english'))
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def clean(text):
    text = text.lower()
    temp = ""
    for i in text.split():
        try:
            temp+=contraction[i]+' '
        except:
            temp+= i+' '
    text = temp.strip()
    text = text.lower().translate(remove_punctuation_map)
    text = re.sub("[^a-zA-Z#]"," ",text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r",", "", text)
    text = re.sub(r"\.", "", text)
    text = re.sub(r"!", "!", text)
    text = re.sub(r"\/", "", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", ":", text)
    text = re.sub(r' +',' ',text)

    text = word_tokenize(text)
    text = [i for i in text if i not in stopw]
    return " ".join(text)

X = ["First came the Golden Age and the Silver Age.",
     "Then came the Bronze Age.",
     "Then was the age of Revolution with the Iron Age.",
     "Now we are in the Digital Age"
     ]

Cleaned_X = [clean(i) for i in X]

dic = list(set(" ".join(Cleaned_X).split()))
feature_names = np.array(dic)

vectorizer1 = CountVectorizer()
M1 = vectorizer1.fit_transform(Cleaned_X)

matrix1 = M1.toarray()
matrix1 = matrix1.astype('float')
feature_names1 = np.array(vectorizer1.get_feature_names()).reshape(1,-1)

tf = np.empty([matrix1.shape[0],matrix1.shape[1]],dtype='float')
tf = tf.astype('float')

for row in range(matrix1.shape[0]):
    l = len(Cleaned_X[row].split())
    for col in range(matrix1.shape[1]):
        temp = (matrix1[row,col]/l)
        tf[row,col] = temp
    
vectorizer2 = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True)
M2 = vectorizer2.fit_transform(Cleaned_X)
feature_names1 = np.array(vectorizer1.get_feature_names()).reshape(1,-1)
vectorizer2.g
matrix2 = M2.toarray()