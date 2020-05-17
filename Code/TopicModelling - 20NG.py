from TextPreprocessing import clean
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
import umap
import matplotlib.pyplot as plt


dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
classes = list(dataset.target_names)

def LSA(k, X):
    vectorizer = TfidfVectorizer(smooth_idf=True, max_features=1000)
    X_transformed = vectorizer.fit_transform(X)
    svd = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=300, random_state=122)
    lsa = svd.fit_transform(X_transformed) 
    document_topic_matrix = pd.DataFrame(lsa,columns=classes, index = X)
    dic = vectorizer.get_feature_names()
    term_topic_matrix = pd.DataFrame(svd.components_, index = classes, columns = (dic)).T
    return lsa, document_topic_matrix, term_topic_matrix

def NMFFunction(k, X):
    vectorizer = TfidfVectorizer(smooth_idf=True, max_features=1000)
    X_transformed = vectorizer.fit_transform(X)
    model = NMF(n_components=k, random_state=122, init='nndsvd')
    topicmodel = model.fit_transform(X_transformed) 
    document_topic_matrix = pd.DataFrame(topicmodel,columns=classes, index = X)
    dic = vectorizer.get_feature_names()
    term_topic_matrix = pd.DataFrame(model.components_, index = classes, columns = (dic)).T
    return topicmodel, document_topic_matrix, term_topic_matrix


df = pd.DataFrame()
df["Topic"] = dataset.target
df["Topic"] = df["Topic"].apply(lambda x: classes[x])
df["Document"] = dataset.data
df["Cleaned Document"] = df["Document"].apply(lambda x: clean(x))

lsa, document_topic_matrix_lsa, term_topic_matrix_lsa = LSA(20, df["Cleaned Document"]) 
nmf, document_topic_matrix_nmf, term_topic_matrix_nmf = NMFFunction(20, df["Cleaned Document"]) 

embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(lsa)
plt.figure(figsize=(7,5))
plt.title("LSA Topic Modelling")
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c = dataset.target,s = 10, edgecolor='none',cmap='plasma')
plt.legend(handles=scatter.legend_elements()[0],labels = classes)
plt.savefig("Topic Modelling LSA.svg",dpi=2000)
plt.show()

embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(nmf)
plt.figure(figsize=(7,5))
plt.title("NMF Topic Modelling")
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c = dataset.target,s = 10, edgecolor='none',cmap='plasma')
plt.legend(handles=scatter.legend_elements()[0],labels = classes)
plt.savefig("Topic Modelling NMF.svg",dpi=2000)
plt.show()