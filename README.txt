Vaibhav Devekar (devekar.1@osu.edu)
Akshay Nikam (nikam.5@osu.edu)


==========================
Data Mining - Assignment 2
==========================

NOTE: The program depends on feature vectors by Assignment 1 program - DM1.py
      DM1.py makes use of NLTK for stemming, hence cannot be run on STDLINUX. 
      Use "make copy" to get the vectors.                                      


Executing the program
---------------------
1) Run command "make preprocess" to generate the feature vectors.
   OR
   Run command "make copy" to copy the vectors and reuters from ~/home/8/devekar/WWW/DM2 (This will not run DM1.py)
2) Run command "make bayesian" to run the Naive Bayesian classifier (80/20 split; Approximate execution time: 5 sec).
3) Run command "make knn" to run the K Nearest Neighbor classifier (99/1 split; K=5; Approximate execution time: 5 min).
4) Run command "make clean" to remove output files.


With parameters
----------------
1) DM2_bayesian.py can also perform cross-validation or run with user-specified training/test split.
   a. For cross-validation, specify -f flag with the number of folds.
      For example, for 5 folds:
          python DM2_bayesian.py -f 5

   b. For custom training/test split, specify -t flag with the testing split size.
      For example, for 60/40 split:
          python DM2_bayesian.py -t 40

2) DM2_KNN.py can be run with custom K and split size.
   For example, for K=5 and a 80/20 split:
   python DM2_KNN.py -k 5 -t 20


Output
------
The output specifies the training and testing time as well as the following metrics: 
Accuracy, Precision, Recall, F-measure, G-mean

The Naive Bayesian Classifier also produces following metrics:
A0: Accuracy by atleast one correct prediction per aticle
    If atleast one topic is predicted correctly, we term the prediction for the article to be successful

A1: Accuracy by all correct predictions per article
    We term the prediction for the article to be successful only if all predictions match with actual topics.

A2: Accuracy by 'correct predictions/total topics' per article
    We specify the success of prediction in a ratio, 3 out of 5 correct predictions means 0.6



Source files
------------
1) DM1.py
2) parse.py
3) preprocess.py
4) DM2_bayesian.py
5) bayesian.py
6) DM2_KNN.py
7) KNN.py
8) stopwords
9) Makefile
 
