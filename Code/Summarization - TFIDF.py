from TextPreprocessing import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np

def sentence_score(X_clean):
    vectorizer = TfidfVectorizer(smooth_idf=True,max_features=1000)
    X = vectorizer.fit_transform(X_clean)
    X = X.toarray()
    ss = [(pos,sum(sentence)) for pos, sentence in enumerate(X)]
    return ss

def summarize_k(k,sentences,scores):
    scores.sort(key = lambda x:x[1], reverse = True)
    scores = scores[:k]
    scores.sort(key = lambda x:x[0])
    doc = [sentences[pair[0]] for pair in scores]
    return " ".join(doc)
    
def summarize_avg(sentences,scores):
    agg = 0
    for i in scores:
        agg+=i[1]
    avg = agg/len(scores)
    scores = [i for i in scores if i[1]>=avg]
    scores.sort(key = lambda x:x[0])
    doc = [sentences[pair[0]] for pair in scores]
    return " ".join(doc)