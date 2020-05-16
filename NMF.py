import pandas as pd;
import numpy as np;
import scipy as sp;
import sklearn;
import sys;
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer;
from sklearn.decomposition import NMF;
from sklearn.preprocessing import normalize;

num_topics = 4;
a1 = "He is a good dog."
a2 = "The dog is too lazy."
a3 = "That is a brown cat."
a4 = "The cat is very active."
a5 = "I have brown cat and dog."
X = [a1,a2,a3,a4,a5]

# The transformation will return a matrix of size (Documents x Features)
# where the value of a cell is going to be the number of times the feature (word) appears 
# in that document.
vectorizer = CountVectorizer(analyzer='word', max_features=1000);
x_counts = vectorizer.fit_transform(X);


# Next, we set a TfIdf Transformer, and transform the counts with the model.
transformer = TfidfTransformer(smooth_idf=False);
x_tfidf = transformer.fit_transform(x_counts);

# To normalize the TfIdf values to unit length for each row.
xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)

#obtain a NMF model.
model = NMF(n_components=num_topics, init='nndsvd');

#fit the model
model.fit(xtfidf_norm)

def get_nmf_topics(model, n_top_words):
    
    #Reverse mapping the work ids to their corresponding words.
    feat_names = vectorizer.get_feature_names()
    word_dict = {};
    for i in range(num_topics):
        
        #for each topic, we obtain the largest values and add the 
        # words they map to into the dictionary.
        words_ids = model.components_[i].argsort()[:-20 - 1:-1]
        words = [feat_names[key] for key in words_ids]
        word_dict['Topic # ' + '{:02d}'.format(i+1)] = words;
    
    return pd.DataFrame(word_dict);

print(get_nmf_topics(model, 20))
