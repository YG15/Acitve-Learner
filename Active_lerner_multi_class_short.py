###############Active_lerner_multi_class############
# Authors: Omri Allouche  & Yonathan Guttel
# Date: 28.12.2017
# Description: Draft work for Text active learner
#####################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc


def get_data_for_model(df, true_label_col='true_label'):
    labels=np.unique(df[true_label_col]) #get uniques labels
    pred_prob_col= [str("prediction_proba_"+x) for x in labels] #create a vector of col names to store each label probability
    cols_to_remove = set(['predictions', pred_prob_col, true_label_col])
    cols_to_keep = [c for c in df.columns if c not in cols_to_remove]
    X = df[cols_to_keep]
    if true_label_col in df.columns:
        y = df[true_label_col]
    else:
        y = None

    return X, y


class ActiveLearner:
    def __init__(self,
                 model,
                 bootstrap_iterations = 0,
                 th_prediction_confidence = 0.8
                ):
        self.bootstrap_iterations = bootstrap_iterations
        self.th_prediction_confidence = th_prediction_confidence
        self.model = model

    def predict(self, df):
        model = self.model
        X, _ = get_data_for_model(df)
        Y_pred = model.predict_proba(X)

        if self.bootstrap_iterations > 0:
            org_df = df.copy()

            for i in range(self.bootstrap_iterations):
                df = org_df[Y_pred > self.th_prediction_confidence]
                X, _ = get_data_for_model(df)
                model.fit(X, Y_pred)
                Y_pred = model.predict_proba(org_df)

            df = org_df

        labels = np.unique(df[true_label_col])
        pred_prob_col = [str("prediction_proba_" + x) for x in labels]

        df['predictions'] = np.argmax(Y_pred, axis=1)
        for col, i in zip(pred_prob_col, range(len(labels))):
            df[col]= Y_pred[i]

        return df

    def fit(self, df):
        X, y = get_data_for_model(df)
        self.model.fit(X, y)
        return self

    def get_next_batch_for_labeling(self, df, batch_size=10):
        X, _ = get_data_for_model(df)
        df = self.predict(X)
        sorted = df.sort_values('predictions_proba')  # sort rows by confidence
        return sorted.head(batch_size)

    """
    Q:
    if i do not use the calss (self element) should i move the method/function outside of the class?
    """

    def oracle_queary (df, true_label_col='true_label'):
        current_batch=get_next_batch_for_labeling(df, batch_size=10) #estract the x samples with the lowest pred_prob values
        for i in range(batch_size):
            print ("Please type the correct label of the following sample: ", end="")
            print (current_batch[i])
            oracle_answer=input("Label: ")
            print ("Your answer is ", oracle_answer, "do you wish to proceed or to correct yor answer?") #verify that the right answer was inputed
            correct_answer = input ("""If label is correct type "Y" if you wish to correct it type "N" """)
            if not correct_answer in ["Y","y","N","n"]:
                correct_answer = input("""Please answer again: if label is correct type "Y", if you wish to correct it type "N" """)
            if correct_answer in ["N","n"]:
                oracle_answer = input("Correct Label: ")

            current_batch[i,true_label_col]=oracle_answer # inter the input to the right label column

        df = df.combine_first(current_batch) #merge the results of new trur label back to the original df

        return df

    def performance_analysis(df, , true_label_col='true_label'):

        org_df = df.copy()
        df_sub = org_df[org_df[true_label_col] != None] # keep only sample which have true label
        labels = np.unique(df_sub[true_label_col]) #see above
        pred_prob_col = [str("prediction_proba_" + x) for x in labels] #see above
        Y_pred = df_sub[pred_prob_col]  # All the diffrent label prediction probabilities columns
        y_true = df_sub[true_label_col] # The labels column
        # Use label_binarize to be multi-label like settings
        Y_true = label_binarize(y_true, classes=labels) #binarize Labels column
        n_classes = Y.shape[1]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_true[:, i], Y_pred[:, i])
            average_precision[i] = average_precision_score(Y_true[:, i], Y_pred[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
        average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision["micro"]))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ROC~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_true[:, i], Y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc["micro"])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


    def exit_and_save(self, df):
        save_answer = input("""Do you wish to save the process? If yes type "Y" """)
        exit_answer=input("""Do you wish to stop? If yes type "Y" """)
        if exit_answer in ["Y","y"] and save_answer in ["Y","y"]:
            #save df and self/model and exit
        elif save_answer in ["Y","y"]:
            # save df and self/model  and proceed
        elif exit_answer in ["Y","y"]:
            # exit without saving
        else
            pass
    return (pass)



