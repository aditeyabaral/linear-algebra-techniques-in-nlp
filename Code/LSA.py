import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from TextPreprocessing import clean

dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))
classes = list(dataset.target_names)
df = pd.DataFrame()
df["Topic"] = dataset.target
df["Topic"] = df["Topic"].apply(lambda x: classes[x])
df["Document"] = dataset.data
df["Cleaned Document"] = df["Document"].apply(lambda x: clean(x))

vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True, max_features=1000)
X = vectorizer.fit_transform(df["Cleaned Document"])
X = X.toarray()

svd = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122) #We know 20 news groups
lsa = svd.fit_transform(X)

dic = vectorizer.get_feature_names()
document_topic_matrix = pd.DataFrame(lsa,columns=classes) #document-topic matrix
document_topic_matrix["Document"] = df["Document"]
term_topic_matrix = pd.DataFrame(svd.components_, index = classes, columns = (dic)).T #term topic matrix


embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(lsa)
plt.figure(figsize=(7,5))
plt.scatter(embedding[:, 0], embedding[:, 1], c = dataset.target,s = 10, edgecolor='none')
plt.show()


'''
terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:8]
    print("Topic:", classes[i])
    for t in sorted_terms:
        print(t[0],end =" ")
    print("\n")    
'''

'''
for topic in classes[11:20]:
    documents = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'),categories=[topic])
    content = documents.data
    cleaned_content = [clean(i) for i in content]
    vectorizer = TfidfVectorizer(lowercase=True, use_idf=True, smooth_idf=True)
    M = vectorizer.fit_transform(cleaned_content)
    matrix = M.toarray()
    lsa = TruncatedSVD(n_components=1, algorithm='randomized', n_iter=100, random_state=0)
    lsa.fit_transform(M)
    terms = vectorizer.get_feature_names()
    print("Topic:",topic)
    for comp in lsa.components_:
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
        for t in sorted_terms:
            print(t[0], end = " ")
        print()
    print()
'''