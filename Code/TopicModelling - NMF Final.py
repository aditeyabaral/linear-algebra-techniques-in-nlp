import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from TextPreprocessing import clean

def NMFFunction(k, X):
    X_clean = list(map(clean,X))
    vectorizer = TfidfVectorizer(smooth_idf=True)
    X_transformed = vectorizer.fit_transform(X_clean)
    model = NMF(n_components=k, random_state=122, init='nndsvd')
    topicmodel = model.fit_transform(X_transformed) 
    column_names = ["Topic {}".format(str(i+1)) for i in range(k)]
    document_topic_matrix = pd.DataFrame(topicmodel,columns=column_names, index = X)
    dic = vectorizer.get_feature_names()
    term_topic_matrix = pd.DataFrame(model.components_, index = column_names, columns = (dic)).T
    return topicmodel, document_topic_matrix, term_topic_matrix
    
a1 = "He is a good dog."
a2 = "The dog is too lazy."
a3 = "That is a brown cat."
a4 = "The cat is very active."
a5 = "I have brown cat and dog."
X = [a1,a2,a3,a4,a5]


matrix, document_topic, term_topic = NMFFunction(2, X)