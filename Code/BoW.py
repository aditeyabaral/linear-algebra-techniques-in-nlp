import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from TextPreprocessing import clean

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
matrix2 = M2.toarray()