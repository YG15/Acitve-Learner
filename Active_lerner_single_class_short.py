###############Active_lerner_single_class############
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
    cols_to_remove = set(['predictions', 'predictions_proba', true_label_col])
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

        df['predictions'] = np.argmax(Y_pred, axis=1)
        df['predictions_proba'] = [x[v] for x, v in zip(Y_pred, df['predictions'])]

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
            print ("Your answer is ", oracle_answer, "do you wish to proceed or to correct your answer?") #verify that the right answer was inputed
            correct_answer = input ("""If label is correct type "Y" if you wish to correct it type "N" """)
            if not correct_answer in ["Y","y","N","n"]:
                correct_answer = input("""Please answer again: if label is correct type "Y", if you wish to correct it type "N""")
            if correct_answer in ["N","n"]:
                oracle_answer = input("Correct Label: ")

            current_batch[i,true_label_col]=oracle_answer # inter the input to the right label column

        df = df.combine_first(current_batch) #merge the results of new trur label back to the original df

        return df

    def performance_analysis(df, , true_label_col='true_label'):

        org_df = df.copy()
        df_sub = org_df[org_df[true_label_col] != None] # keep only sample which have true label
        y_pred = df_sub["predictions_porba"]
        y_true = df_sub[true_label_col] #what should it be?
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # For each class
        precision, recall, _ = precision_recall_curve(y_true,y_pred)
        average_precision= average_precision_score(y_true, y_pred)

        #plotting
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision["micro"]))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ROC~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
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
        return(pass)
