from TextPreprocessing import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
import pandas as pd

def LSA(k, X):
    X_clean = list(map(clean,X))
    vectorizer = TfidfVectorizer(smooth_idf=True, max_features=1000)
    X_transformed = vectorizer.fit_transform(X_clean)
    feature_count = len(vectorizer.get_feature_names())
    if k>feature_count:
        k = feature_count-1
    svd = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=300, random_state=122)
    lsa = svd.fit_transform(X_transformed)
    Sigma = svd.singular_values_
    column_names = ["Topic {}".format(str(i+1)) for i in range(lsa.shape[1])]
    document_topic_matrix = pd.DataFrame(lsa,columns=column_names)
    document_topic_matrix["Document"] = X
    document_topic_matrix["Position"] = [i for i in range(len(X))]
    dic = vectorizer.get_feature_names()
    term_topic_matrix = pd.DataFrame(svd.components_, index = column_names, columns = (dic)).T
    return k,Sigma, lsa, document_topic_matrix, term_topic_matrix

def percent_sigma(Sigma, X):
    agg = sum(Sigma)
    return (Sigma/agg)*len(X)

def summarize(k,sigma, document_term):
    summary = []
    column_names = ["Topic {}".format(str(i+1)) for i in range(len(sigma))]
    for i in range(len(column_names)):
        num_sent = sigma[i]
        topic = column_names[i]
        document_term.sort_values(by = topic,inplace=True,ascending = False)
        
        document_term.reset_index(inplace=True)
        document_term.drop(columns = ["index"],inplace=True)
        for j in range(num_sent):
            item = (document_term["Document"][j],document_term["Position"][j])
            if item not in summary:
                summary.append(item)
    summary.sort(key = lambda x: x[1])
    sent = [i[0] for i in summary]
    return " ".join(sent)
