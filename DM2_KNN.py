#!/usr/bin/python
import csv
from KNN import KNN
#from naive_bayesian_classify import NaiveBayesianClassifier

def parseDM(filepath = r'data_matrix.csv'):
    dataMatrix = []

    matrix = []
    word_list = []
    topic_list = []
    with open(filepath, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        for row in reader:
            dataMatrix.append(row)

    for item in dataMatrix[1]:
        if "_" not in item:
            word_list.append(item)
        elif "t_" in item:
            topic_list.append(item[2:])

    word_list = word_list[1:] # Remove 'Article #'
    words_topics_size = len(topic_list) + len(word_list)

    for row in dataMatrix[2:]:
        matrix.append( [row[0]] + map(int, row[1:1 + words_topics_size]) )
    return {"topic_list":topic_list, "word_list": word_list, "matrix": matrix}


dataMatrix = parseDM()
#nbc = NaiveBayesianClassifier()
#nbc.classify(datamatrix)

k = 10
knn = KNN(dataMatrix, k)
knn.test_split(1)
