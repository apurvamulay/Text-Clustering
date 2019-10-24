## Based on Python 2.7 and executed on Pycharm

import re
import operator
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

# Read stop-words file, stripe it to remove \n and store it in stopWords
stopWords_file = open("stopWords.txt", "r", )
stopWords = stopWords_file.readlines()
stopWords = map(lambda s: s.strip(), stopWords)


# Define functions to remove html characters and punctuations from each line
def remove_html(sentence):
    regex = re.compile('<.*?>')
    clean_text = re.sub(regex, ' ', sentence)
    return clean_text


def remove_punctuation(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)(|\|/]', r' ', cleaned)
    return cleaned


# Open finefoods.txt file and read it in lines variable
text_file = open("finefoods.txt", "r")
lines = text_file.readlines()

# Create empty reviews array and initialize empty counter
reviews = []
word_freq = Counter()

# Iterate over the lines, split, remove html and punctuation characters, add to reviews array
# Update frequency of words. This is will give unique words as counter will create dictionary
for line in lines:
    data = line.split('review/text:')
    if len(data) > 1:
        review = data[1]
        review = review.lower()
        review = remove_html(review)
        review = remove_punctuation(review)
        reviews.append(review)
        reviewWords = review.split()
        word_freq.update(reviewWords)  # L= word_freq

# Stopwords are extended as it contains below extra characters
stopWords.extend(
    ['-', '2', '1', '3', '5', '4', '6', '--', '12', '10', '8', ':', '50', '7', '20', '&', '24', '9', 'will', 'dont',
     'didnt', 'cant', 'doesnt', 'isnt', 'ive'])

# Remove stopwords from unique words
for wordsToRemove in stopWords:
    word_freq.pop(wordsToRemove, None)  # W

# to get top 500 words, sort the word_freq array in descending order of frequencies
top_500_words_map = sorted(word_freq.items(), key=operator.itemgetter(1), reverse=True)[:500]  # 500 words
print('top_500_words_map', top_500_words_map)

top_500_words = [x[0] for x in top_500_words_map]

file_center = open('top_500_words.txt', 'w')
file_center.write(str(top_500_words_map))
file_center.close()

# Vectorize the reviews using TFIDF vectorization as it reduces bias compared to count vectorization
# Encoding is added a parameter to vectorizer as it throws encoding error
vectorizer = TfidfVectorizer(encoding='latin-1')
vectorizer.fit(top_500_words)
vector = vectorizer.transform(reviews)

# Using MiniBatchKmeans as it is faster than KMeans.
# Random state is added to get same clusters every time.
model = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=1000, random_state=101)
model.fit(vector)

print('Cluster Centroids:')
print(model.cluster_centers_)


print('Top terms per cluster:')
# cluster centroids after ordering
order_centroids = model.cluster_centers_.argsort()[:,::-1]  # sorts all words in descending order based on cluster centers
terms = vectorizer.get_feature_names()  # feature names = top 500 words

print("order_centroids", order_centroids)

topWords_centroids = open('topWords_centroids.txt', 'w+')

topWords = []
for i in range(10):
    print("Cluster %d:" % i),
    topWordsWithinCluster = []
    for index in order_centroids[i, :5]:  # Give top 5
        print(str(terms[index]), ' %s ' % model.cluster_centers_[i][index])  # words with its centroids for every cluster
        word_centroid_tuple = ' ' + str(terms[index]) + ' : ' + str(model.cluster_centers_[i][index]) + ' '
        topWords_centroids.write(str(word_centroid_tuple))
        topWordsWithinCluster.append(terms[index])
    topWords.append(topWordsWithinCluster)  # list of top words of every cluster

review_cluster = list(zip(reviews, model.labels_))  # cluster to which a particular review is assigned

# Dataframe which stores which review is mapped to which cluster
review_cluster_df = pd.DataFrame(review_cluster, columns=["Review", "Cluster"])
print(review_cluster_df.head())  # return 5 rows as 5 is default value

review_per_cluster = open('review_per_cluster.txt', 'w')
review_per_cluster.write(str(review_cluster_df))
review_per_cluster.close()

# Get the cluster distribution
print('Cluster Distribution: ')
for i in range(10):
    print(review_cluster_df[review_cluster_df['Cluster'] == i]['Review'].count())


# function to get frequencies in each cluster
def get_freq_each_cluster(corpus):
    freq = Counter()
    for line in corpus:
        words = line.split()
        freq.update(words)
    return freq


test_output = open('test_output.txt', 'w')

print("Frequency of top 5 words in each cluster")
for i in range(10):
    print("Cluster %d:" % i)
    reviewInEachCluster = review_cluster_df[review_cluster_df['Cluster'] == i]['Review']
    wordFrequency = get_freq_each_cluster(reviewInEachCluster)
    for j in range(5):
        currentWord = topWords[i][j]
        value = currentWord + ': ' + str(wordFrequency[currentWord])
        test_output.write(value)
        print(currentWord + ': ' + str(wordFrequency[currentWord]))  # freq of all words in that review
test_output.close()
