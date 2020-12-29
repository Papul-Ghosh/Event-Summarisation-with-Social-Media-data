# Event-Summarisation-with-Social-Media-data

**Project Summary:**
Applying various machine learning algorithms to the collected dataset on the Nepal Earthquake incident to analyse their sentiments and clustering the tweets in multiple groups having their own specific categories of objectives to finally extract the summarised information.

**Technology used:** Python.

**Abstract:**

Microblogging sites have millions of people sharing their thoughts daily because of its characteristic short and simple manner of expression. During the sudden onset of a crisis situation, affected people post useful information on Twitter that can be used for situational awareness and other humanitarian disaster response efforts, if processed timely and effectively. Processing social media information pose multiple challenges such as learning information categories from the incoming stream of messages and classifying them into different classes among others. In this project some machine learning algorithms have been used to the collected twitter dataset on the unfortunate Nepal Earthquake incident to analyse the sentiment of each tweet and classify its polarity as positive, negative or neutral. Based on the relative similarities present in the tweets and the calculated sentiments,the tweets have been clustered in multiple groups having their own specific categories of objectives. Thus, summarised information can be extracted from huge number of blogs much efficiently in a faster way that in turn may be fruitful in disaster management.

**Dataset:**

We use the dataset on the disastrous earthquake in Nepal in the year 2015, which is publicly available on crisisnlp.qcri.org. Here is a description of the data, provided by the website:
The set consists of 3000 tweets on the unfortunate Nepal earthquake, specially selected for sentiment analysis between the period 2015-04-25 12:53:13 to 2015-05-19 06:35:49. Each of these rows contains tweet date, tweet category, confidence value of the tweet category, tweet id followed by the tweet itself. Furthermore, the retweets present in the dataset is mentioned with ‘RT’ keyword at the beginning. As per Twitter norms each tweet has maximum length of words.

**Implementation:**

1. Text Preprocessing
2. TF-IDF Score Calculation
3. Cosine Similarity Computation
4. Clustering based on Cosine Similarity Values
5. Sentiment Analysis
6. Interim Data Representation
7. K-Mean Clustering

**Results:**

By clustering the dataset of 3000 tweets using cosine similarity we have obtained 242 distinct clusters. From each cluster we have extracted the most significant one (which have highest TF-IDF score) for further processing. Finally applying KMean clustering on the 242 intermediate tweets based on the TF-IDF metrics and sentiment scores we have grouped the corpus into different number of clusters (i.e. k= 10, 20, 30).

I have clustered the dataset based on the cosine similarity values between each pair of tweets and have implemented k-means clustering and thus extracted the most significant tweets. We have achieved a satisfactory classification with less standard deviation.

Detailed results can be found on final_report.pdf
