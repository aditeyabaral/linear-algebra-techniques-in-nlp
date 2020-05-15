import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from TextPreprocessing import clean


a1 = "He is a good dog."
a2 = "The dog is too lazy."
a3 = "That is a brown cat."
a4 = "The cat is very active."
a5 = "I have brown cat and dog."
X = [a1,a2,a3,a4,a5]

Cleaned_X = list(map(clean,X))
df = pd.DataFrame()
df["Documents"] = X
df["Cleaned"] = Cleaned_X

vectorizer = TfidfVectorizer(stop_words='english', smooth_idf=True, use_idf= True, lowercase=True)
Transform_X = vectorizer.fit_transform(Cleaned_X)
df["Vector"] = Transform_X
Transform_X = Transform_X.toarray()
dic = vectorizer.get_feature_names()

svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=100, random_state=122)
lsa = svd_model.fit_transform(Transform_X) 
document_topic_matrix = pd.DataFrame(lsa,columns=["topic_1","topic_2"]) #document-topic matrix
document_topic_matrix["Documents"] = df["Documents"]
term_topic_matrix = pd.DataFrame(svd_model.components_, index = ["topic_1","topic_2"], columns = (dic)).T #term topic matrix