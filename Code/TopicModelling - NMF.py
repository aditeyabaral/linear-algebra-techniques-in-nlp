import pandas as pd
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from TextPreprocessing import clean

def NMFFunction(k, X):
    vectorizer = TfidfVectorizer(smooth_idf=True, max_features=1000)
    X_transformed = vectorizer.fit_transform(X)
    model = NMF(n_components=k, random_state=122, init='nndsvd')
    topicmodel = model.fit_transform(X_transformed) 
    column_names = ["Topic {}".format(str(i+1)) for i in range(k)]
    document_topic_matrix = pd.DataFrame(topicmodel,columns=column_names, index = X)
    dic = vectorizer.get_feature_names()
    term_topic_matrix = pd.DataFrame(model.components_, index = column_names, columns = (dic)).T
    return topicmodel, document_topic_matrix, term_topic_matrix


''' To display top words per topic
terms = vectorizer.get_feature_names()
for i, comp in enumerate(svd.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:8]
    print("Topic:", classes[i])
    for t in sorted_terms:
        print(t[0],end =" ")
    print("\n")    
'''