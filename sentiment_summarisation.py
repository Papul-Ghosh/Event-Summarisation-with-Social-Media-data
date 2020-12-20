# Importing libraries
import csv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster import  hierarchy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import numpy as np
import scipy
import statistics

stopword = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

#step 1
# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    url_free =' '.join([ch for ch in doc.split() if '.' not in ch and '@' not in ch and "'" not in ch and '"' not in ch and "`" not in ch])
    tokenized_word=word_tokenize(url_free)
    stop_free =[i for i in tokenized_word if i not in stopword if i not in exclude]
    normalized =[lemma.lemmatize(word) for word in stop_free]
    return ' '.join(normalized)

def get_cosine_sim(strs): 
    vectors = get_vectors(strs)
    vectors = vectors.todense()
    threshold = 0.85     #step 4: clustering on cosine similarity
    Z = hierarchy.linkage(vectors,"average", metric="cosine")
    C = hierarchy.fcluster(Z, threshold, criterion="distance")
    return C

def get_vectors(text):
    X = CountVectorizer().fit_transform(text)
   # X = TfidfTransformer().fit_transform(X)
    return(X)


def delete_rows_csr(mat, indices):    #remove the rows denoted by indices form the CSR sparse matrix mat
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.zeros(mat.shape[0], dtype=bool)
    mask[indices] = True
    return mat[mask]



path = 'nepal_eq_2015.csv'

tweetlist= []
dataset=[]

with open(path,'r',encoding='utf8') as csv_file:
    csv_reader = csv.reader(csv_file)
    dataset= list(csv_reader)
del dataset[0]      #delete header row

for row in dataset[:3000]:
    tweetlist.append(clean(row[-1]))        #step 1: extract tweet column

#use print(X[i]) to print tfidf scores of X[i]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(tweetlist)     #step 2: tfidf calculation

'''kmeans = KMeans(n_clusters=10).fit(X)
lines_for_predicting = ["tf and idf is awesome!", "We feel sorry for the dead"]
means.predict(vectorizer.transform(lines_for_predicting))
'''

cos_sim_cluster=get_cosine_sim(tweetlist)      #step 3: cosine similarity calculation

cos_sim_groups=[]
sorted_tweet_index_list=[]

for i in range(1,max(cos_sim_cluster)+1):       #creating groups with same cos_sim_cluster id
    cos_sim_groups.append([j for j,val in enumerate(cos_sim_cluster) if val==i])

for group in cos_sim_groups:            #extracting index of max tfidf score from each group
    tfidf_score= [tfidf[k].sum() for k in group]
    max_tfidf_id= group[tfidf_score.index(max(tfidf_score))]
    sorted_tweet_index_list.append(max_tfidf_id)


#step 5: sentiment analysis
sentiment=[]
sid = SentimentIntensityAnalyzer()
for sentence in tweetlist:
    polarity = sid.polarity_scores(sentence)
    sentiment.append(list(polarity.values()))
#for k in polarity:
#     print(‘{0}: {1}, ‘.format(k, ss[k]), end=’’)
#{'neg': 0.264, 'neu': 0.736, 'pos': 0.0, 'compound': -0.6486}



#step 6: represent as vectors
temp=[]
sorted_tfidf=delete_rows_csr(tfidf, sorted_tweet_index_list)
vector=[]
tweet_id=[]
sorted_sentiment=[]
for row in sorted_tweet_index_list:
    #sorted_tfidf.append(tfidf[row])
    tweet_id.append([int(dataset[row][-2][1:-1])])     #tweet id except ''
    sorted_sentiment.append(sentiment[row])
    

#step 7: cluster using kmeans
#kmeans1 = KMeans(n_clusters=10).fit(sorted_tweet_index_list)
vector= hstack([sorted_tfidf, sorted_sentiment]).toarray()      #combine tfidf scores with tweet id
#vector= np.asarray([sorted_tfidf, tweet_id, sorted_sentiment],dtype=object)
for k in[10,20,30]:
    kmeans = KMeans(n_clusters=k).fit(vector)
    # print using kmeans.labels_
    # print certain cluster using np.where(kmeans.labels_ == 4)

    '''
    #output tweets
    csvData = [['  index  ','  Tweet_id  ', '  Sentiment  ', '  Cluster id  ', '  TFIDF score  ', '  Group number  ','  Tweet  ']]
    for index in range(len(sorted_tweet_index_list)):
        row=[]
        row.append(index+1)
        row.append(tweet_id[index][0])
        row.append(sorted_sentiment[index][0])
        row.append(kmeans.labels_[index])
        row.append(sorted_tfidf[index].sum())
        row.append(cos_sim_cluster[sorted_tweet_index_list[index]])
        row.append(dataset[sorted_tweet_index_list[index]][-1])
        csvData.append(row)

    with open('Nepal_eq.csv', 'w', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
        print('Data printed')


    csvFile.close()
    '''

    #Statistical Table
    kmean_cluster=[]
    for id in range(max(kmeans.labels_)+1):
        kmean_cluster.append([(rindex,sentiment[rindex][-1]) for rindex in sorted_tweet_index_list if kmeans.labels_[sorted_tweet_index_list.index(rindex)]==id])
    
    csvData = [[ '  Cluster id  ', '  Average  ', '  Min  ','  Max  ', '  SD  ', '  Tweet  ']]
    for r in kmean_cluster:
        id= kmean_cluster.index(r)+1
        sent_cluster=[t[1] for t in r]
        mean= sum(sent_cluster)/len(sent_cluster)
        SD= statistics.stdev(sent_cluster)
        minm= min(sent_cluster)
        maxm= max(sent_cluster)
        mid_index=int(len(r)/2)
        final_tweet_index=sorted(r)[mid_index][0]
        tweet=dataset[final_tweet_index][-1]
        csvData.append([id, mean, minm, maxm, SD, tweet])
    
    
    with open('final_table.csv', 'a', encoding="utf-8", newline='') as csvFile:
        writer = csv.writer(csvFile)
        #writer.writerows("#considering",k,"Clusters:")
        writer.writerows(csvData)
        writer.writerows("")
        csvFile.close()


    csv_tfidf = [[ '  Cluster id  ', '  TF-IDF Average  ', '  TF-IDF SD  ']]
    for r in kmean_cluster:
        id= kmean_cluster.index(r)+1
        tfidf_cluster=[tfidf[t[0]].sum() for t in r]
        mean= sum(tfidf_cluster)/len(tfidf_cluster)
        SD= statistics.stdev(tfidf_cluster)
        csv_tfidf.append([id, mean, SD])    
    
    with open('tfidf_table.csv', 'a', encoding="utf-8", newline='') as tfidf_File:
        writer = csv.writer(tfidf_File)
        #writer.writerows("#considering",k,"Clusters:")
        writer.writerows(csv_tfidf)
        writer.writerows("")
        tfidf_File.close()
    print('Data printed',k)

    cluster_id= list(range(1,k+1))
    cluster_size = [len(groups) for groups in kmean_cluster]

    plt.bar(cluster_id, cluster_size, align='center', alpha=0.5)
    plt.xticks(cluster_id)
    plt.xlabel('cluster id')
    plt.ylabel('size of cluster')
    plt.title('Cluster vs cluster size for k='+str(k))
    plt.show()
    
print('All printed ')
