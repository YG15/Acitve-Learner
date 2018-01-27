################################################
# Name: Multinominal Naive Bayes
# Author: Yonathna Guttel & Omri Allouche
# Purpose: Run a Multinominal Naive Bayes
#          algorithms on labeled text which belongs to different classes
# Arguments: A corpus (or Corpora) in a single file, can be changed but
#            than the upload and preprocessing method should be corrected to match the new data
# Returning: Prediction labels vector and accuracy measurement
# date: 14.01.2018
# Version: 2
##############################################
"""
Future plans:
The Script should be organized in several levels:
1. Most of the functions should be gathered under a new class "mulitnomial_NB_classifier"
2. Improvement of the code complexity should be done in order to decrease it and reduce the running time of the algorithm
"""

import pandas as pd
import numpy as np
import os
from copy import deepcopy
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import label_binarize
# from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from collections import Counter
from itertools import compress


def get_data(file_name, binerize=False):
    # The functions upload a raw text file in which each line is a text sample, where the first line is tit class,
    # separated from the text which triple blnk spaces. the function upload it as a pandas df, with two columns;
    # "labels" and text". If the samples are kept in another format the proper changes should be done in the code,
    # in order to ensure a working script

    df = pd.read_csv(file_name, delimiter="\t", header=None, names=["label", "text"])
    target = df['label']
    if binerize == True:  # in case we want to binarize results
        classes = np.unique(target)
        target = label_binarize(target, classes=classes)
    data = df.iloc[:, -1]
    return (data, target)


def tokenize(text):
    tokened_text = [word_tokenize(x) for x in text]
    # stemmer = PorterStemmer()
    # stemmed_text = [stemmer.stem(x).lower()) for x in tokened_text]  #Is it needed?
    return (tokened_text)  # return(stemmed_text)


def priors(labels):  # set the priors of the classes for the MNB classifier
    classes_names = np.unique(labels)
    n_class = len(classes_names)
    priors = {}
    for c in classes_names:
        priors[c] = (Counter(labels)[c]) / n_class
    return (priors)


def likelihoods(tokened_text, labels, smoothing_val=1, test_text=[]):  # tokens probabilities
    classes_names = np.unique(labels)
    classes_data = {}
    likelihood = defaultdict(dict)
    combined_text = tokened_text + test_text  # This is done to prevent a bug in which during prediction step a show up in the test a word which does not have a measured probability
    voc_words = set([item for sublist in combined_text for item in sublist])
    voc_size = len(voc_words)
    for c in classes_names:
        classes_data[c] = list(compress(tokened_text, list(labels == c)))
        flat_list = [item for sublist in classes_data[c] for item in sublist]
        flat_list_length = len(flat_list)
        for w in voc_words:
            likelihood[c][w] = (flat_list.count(w) + smoothing_val) / (
                    flat_list_length + voc_size)  # claculating probability for each words in each class
    return (likelihood)


def MNB_train(text, labels, smoothing_val, test_text=[]):
    tokened_text = tokenize(text)
    prior = priors(labels)
    likelihood = likelihoods(tokened_text, labels, smoothing_val=smoothing_val, test_text=test_text)
    return (prior, likelihood)


def MNB_predict(text, labels, test, raw=True, smoothing_val=1):
    class_names = np.unique(labels)
    final_labels = []
    prop_test = deepcopy(test)
    if raw == True:
        prop_test = tokenize(test)
    prior, likelihood = MNB_train(text, labels, smoothing_val, test_text=prop_test)
    for t in prop_test:
        class_scores = {}
        for c in class_names:
            class_scores[c] = prior[c] * np.prod(np.array([likelihood[c][w] for w in t]))
        final_labels.append(max(class_scores, key=class_scores.get))
    return (final_labels)


# Test our results
if __name__ == '__main__':
    # Set the working directory
    path = "F:\Guttel\Desktop"
    os.chdir(path)

    # Set parameters
    alpha = 1
    test_p = 0.2

    # Choose the corpus on which the model will run
    file_name = 'SMSSpamCollection.txt'

    # Preprocess it for the NB
    X, y = get_data(file_name)
    test_sep_ind = round(len(y) * test_p)

    # present results
    MNB_pred = MNB_predict(X[test_sep_ind:], y[test_sep_ind:], X[:test_sep_ind], raw=True, smoothing_val=alpha)
    true = y[:test_sep_ind]

    MNB_accuracy_score = accuracy_score(true, MNB_pred)
    print("Multinomial accuracy score is", "{0:.4f}".format(MNB_accuracy_score))