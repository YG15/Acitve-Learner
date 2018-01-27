<<<<<<< HEAD
import unittest
from  NB_yonathan import MNB_predict
import NB_yonathan.get_data
import NB_test_function.orgenize_data
import NB_test_function.get_data


class MyTestCase(unittest.TestCase):
    # set the working directory
    path = "F:\Guttel\Desktop"
    os.chdir(path)
    alpha = 1
    test_p = 1000

    # Choose the corpus on which the model will run
    file_name = 'SMSSpamCollection.txt'

    # Preprocess it for the NB
    X_ref, y_ref = NB_test_function.get_data(file_name)
    X_tested, y_tested = NB_yonathan.get_data(file_name)
    true = y[:test_p]

    # present results
    pred_tested = MNB_predict(X[test_sep_ind:], y[test_sep_ind:], X[:test_sep_ind], raw=True, smoothing_val=alpha)
    tested_accuracy_score = accuracy_score(true, MNB_pred)
    ref_clf = MultinomialNB()
    ref_clf.fit(X[test_p:], y[test_p:])
    pred_ref = (MNB_clf.predict(X[:test_p]))
    ref_accuracy_score = MNB_clf.score(X[:test_p], y[:test_p])


    def test_something(self):
        self.assertEqual(tested_accuracy_score, ref_accuracy_score)

if __name__ == '__main__':

    unittest.main()
=======
################################################
# Name: Multinominal Naive Bayes test unit
# Author: Yonathna Guttel & Omri Allouche
# Purpose: Run a unit test for the Multinominal Naive Bayes
#          algorithms written by YG & OA
# Arguments: A corpus (or Corpora) in a single file,
#           and two test functions NB_yonathan & NB_test_function
# Returning: Unit test answer- Ok or not
# date: 14.01.2018
# Version: 1
##############################################

#import needed packages
import unittest
import os
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

# set the working directory (needed for the following imports)
path = "F:\Guttel\Desktop\selfNB"
os.chdir(path)

#import needed funcction
from NB_yonathan import MNB_predict
from NB_yonathan import get_data
from NB_test_function import ref_orgenize_data
from NB_test_function import ref_get_data


class MyTestCase(unittest.TestCase):

    #calculate pred ac
    alpha = 1
    test_p = 1000

    # Choose the corpus on which the model will run
    file_name = 'SMSSpamCollection.txt'

    # Data retrieval
    X_ref, y_ref = ref_get_data(file_name)
    X_pred, y_pred = get_data(file_name)
    true = y_ref[:test_p]

    # Training and Prediction for both functions
    pred_tested = MNB_predict(X_pred[test_p:], y_pred[test_p:], X_pred[:test_p], raw=True, smoothing_val=alpha)
    pred_accuracy_score = accuracy_score(true, pred_tested)
    ref_clf = MultinomialNB()
    ref_clf.fit(X_ref[test_p:], y_ref[test_p:])
    pred_ref = (ref_clf.predict(X_ref[:test_p]))
    ref_accuracy_score = ref_clf.score(X_ref[:test_p], y_ref[:test_p])

    #testing results similarity
    def test_something(self,pred_accuracy_score=pred_accuracy_score, ref_accuracy_score=ref_accuracy_score):
        self.assertAlmostEqual(pred_accuracy_score, ref_accuracy_score, places=1)
        #could be tested on prediction asrrays directly and not scores
        #consider to change the type or parameter of the assertion

if __name__ == '__main__':

    unittest.main()
>>>>>>> a93f65d3b5c8238491d6fc64331e6ea167a2d1f1
