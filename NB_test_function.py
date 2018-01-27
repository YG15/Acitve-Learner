################################################
# Name: Multinominal Naive Bayes test function
# Author: Yonathna Guttel & Omri Allouche
# Purpose: Run a Multinominal Naive Bayes
#          algorithms on labeled text which belongs to different classes
#          based on MNB function from sk.learn
# Arguments: A corpus (or Corpora) in a single file, can be changed but
#            than the upload and preprocessing method should be corrected to match the new data
# Returning: Prediction labels vector and accuracy measurement
# date: 14.01.2018
# Version: 1
##############################################
import pandas as pd
import numpy as np
import os
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.preprocessing import label_binarize


def orgenize_data(data):
    """
    The functions takes a df/array of a raw texts and transform
    them into a tf-idf objects array which can use as the x for a NB algorithm
    """
    raw_df = deepcopy(data)  # copy data
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(
        raw_df)  # Text preprocessing, tokenizing and filtering of stopwords - we build a dictionary of features and transform documents to feature vectors
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(
        X_train_counts)  # Calculating "Term Frequency times Inverse Document Frequency" (tf=idf)
    return (X_train_tfidf)


def get_data(file_name, binerize == False):
    """
    The functions upload a raw text file in which each line is a text sample, where the first line is tit class,
    separated from the text which triple blnk spaces. the function upload it as a pandas df, with two columns;
    "labels" and text". If the samples are kept in another format the proper changes should be done in the code,
    in order to ensure a working script
    """
    df = pd.read_csv(file_name, delimiter="\t", header=None, names=["label", "text"])
    data = orgenize_data(df['text'])
    target = df['label']
    if binerize == True:
        classes =np.unique(df['label'])
        target = label_binarize(df['label'], classes=classes)

    return (data, target)


    # Set the classifier and fit it to thw data
    # Here is the place for later stages to set the classifiers parapmeters
    MNB_clf = MultinomialNB()

    # fir models
    MNB_clf.fit(X[100:], y[100:])

    # present results
    MNB_pred = (MNB_clf.predict(X[:100]))
    MNB_accuracy_score = MNB_clf.score(X[:100], y[:100])
