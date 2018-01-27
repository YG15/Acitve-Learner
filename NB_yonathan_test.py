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
