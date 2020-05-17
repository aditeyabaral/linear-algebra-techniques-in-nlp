from TextPreprocessing import clean
import SummarizationLSA
import SummarizationTFIDF
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
from nltk.book import *
import pandas as pd
import numpy as np

documents = [text1,text2,text3,text4,text5,text6,text7,text8,text9]
documents = [" ".join(i.tokens) for i in documents]

r = Rake()
keyvalues = []
ctr = 1
for doc in documents:
    print(ctr)
    r.extract_keywords_from_text(doc)
    rake_count_total = len(r.get_ranked_phrases())
    
    X = sent_tokenize(doc)
    X_clean = list(map(clean,X))
    
    sentence_scores = SummarizationTFIDF.sentence_score(X_clean)
    summary1 = SummarizationTFIDF.summarize_avg(X,sentence_scores)
    r.extract_keywords_from_text(summary1)
    rake_count_tfidf = len(r.get_ranked_phrases())
    
    k = 200
    new_k, Sigma, lsa, document_topic_matrix, term_topic_matrix = SummarizationLSA.LSA(k,X)
    percentage_topic = SummarizationLSA.percent_sigma(Sigma,X).astype(int)
    summary2 = SummarizationLSA.summarize(new_k,percentage_topic,document_topic_matrix)
    r.extract_keywords_from_text(summary2)
    rake_count_lsa = len(r.get_ranked_phrases())
    
    t = (rake_count_tfidf/rake_count_total,rake_count_lsa/rake_count_total)
    print(t)
    keyvalues.append(t)
    ctr+=1
    
df = pd.DataFrame(keyvalues,columns = ["TF-IDF","LSA"])

