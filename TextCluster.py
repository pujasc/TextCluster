# 
# This script finds cluster based on the textual description provided. 
#

import pandas as pd
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

#reads the csv data file to a datframe 
def readdata(filename):
    try:
        data = pd.read_csv(filename,encoding='cp850')
        return data
    except IOError:
        print ("Error: can\'t find file or read data")

#preprocessing done using regex on the column name meentioned in the call to this function
def preprocessing(data,description):
    data['feature1'] = data[description].replace('[^a-zA-Z]', ' ', regex=True)
    data['feature1'] = data['feature1'].replace(' +', ' ', regex=True)
    return data

#Converts thee text to a vector using TF_Idf vectorizer
def vectorization(data):
    vectorizer = TfidfVectorizer(analyzer='word')
    tf_idf = vectorizer.fit_transform(data['feature1'])
    return tf_idf,vectorizer

#calls the mini batch KMeans
def model(num_cluster,vector):
    kmeans = MiniBatchKMeans(n_clusters=num_cluster).fit(vector)
    return kmeans

#Derive the top 10 words from each cluster
def top_words(vectorizer,model,num_cluster=8):
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    words = ''
    for i in range(num_cluster):
        words += "\nCluster "+str(i)+": \n" 
        for ind in order_centroids[i, :20]:
            words +=  str(terms[ind])+", "
    #print(words)
    with open("TopWords.txt", "w") as text_file:
        text_file.write(words)

# base function
def predict_cluster(filename,description,num_cluster):
    Data = readdata(filename)
    ProcessedData = preprocessing(Data,description)
    tf_idf,vectorizer = vectorization(ProcessedData)
    prediction = model(num_cluster,tf_idf)
    Data['Prediction'] = prediction.labels_
    Data.to_csv('ClusteringResult.csv')
    print("The prediction is stored as ClusteringResult.csv file in the current directory.")
    top_words(vectorizer,prediction,num_cluster)
    print("The top words from each cluster is stored in TopWords.txt file in the current directory.")
    return None


if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        print("Please enter the name of your csv file")
    else:
        predict_cluster(filename,'DESCRIPTION',8)
        # change the number here as per the number of clusters you need.

