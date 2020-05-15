from TextPreprocessing import clean
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import pandas as pd
import os

def TF(word, document):
    return document.count(word)

def TF_scaled(word, document, documents):
    return document.count(word)/len(documents)

def df(documents,term):
    ctr = 0
    for doc in documents:
        if term in doc:
            ctr+=1
    return ctr

def idf(documents,term):
    n = len(documents)
    return 1+np.log((1+n)/(1+df(documents,term)))

def TFIDF(documents):
    preprocessed = list(map(clean, documents))
    vocabulary = list(set(" ".join(preprocessed).split()))
    vocabulary.sort()
    X = np.zeros([len(vocabulary), len(documents)],dtype = 'float')
    idf_scores = [idf(preprocessed,i) for i in vocabulary]
    print(idf_scores)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = TF_scaled(vocabulary[i], preprocessed[j], preprocessed)*idf(preprocessed,vocabulary[i])
    return X
    
    
def BagOfWords(documents):
    preprocessed = list(map(clean, documents))
    vocabulary = list(set(" ".join(preprocessed).split()))
    vocabulary.sort()
    X = np.zeros([len(vocabulary), len(documents)],dtype = 'float')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i,j] = TF(vocabulary[i], preprocessed[j])
    return X


            
documents = ["First came the Golden Age and the Silver Age.",
     "Then came the Bronze Age.",
     "Then was the age of Revolution with the Iron Age.",
     "Now we are in the Digital Age"
     ]

path = os.getcwd()+"\\Reviews"
files = os.listdir(path)
files = [open(path+"\\"+i,"r",encoding='utf-8').read() for i in files]
Clean_files = list(map(clean,files))

vectorizer1 = CountVectorizer(lowercase=True, max_features=1000)
M1 = vectorizer1.fit_transform(Clean_files)
matrix1 = M1.toarray().T

vectorizer2 = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True, max_features=1000)
M2 = vectorizer2.fit_transform(Clean_files)
matrix2 = M2.toarray().T

tf = pd.DataFrame(matrix1,index=vectorizer1.get_feature_names())
tfidf = pd.DataFrame(matrix2,index=vectorizer2.get_feature_names())
