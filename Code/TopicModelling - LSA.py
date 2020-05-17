import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from TextPreprocessing import clean

def LSA(k, X):
    X_clean = list(map(clean,X))
    vectorizer = TfidfVectorizer(smooth_idf=True)
    X_transformed = vectorizer.fit_transform(X_clean)
    svd = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=300, random_state=122)
    lsa = svd.fit_transform(X_transformed) 
    column_names = ["Topic {}".format(str(i+1)) for i in range(k)]
    document_topic_matrix = pd.DataFrame(lsa,columns=column_names, index = X)
    dic = vectorizer.get_feature_names()
    term_topic_matrix = pd.DataFrame(svd.components_, index = column_names, columns = (dic)).T
    return lsa, document_topic_matrix, term_topic_matrix
    

