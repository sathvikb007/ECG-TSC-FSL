import numpy as np
import pandas as pd
import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import utils

class ED_Model:
    def __init__(self):
        self.ED_accs = []
        self.ED_recalls = []
        self.ED_precisions = []
        self.ED_F1s = []

    def get_metrics_of_task_ED(self, support_set, support_labels, query_set, query_labels):
        """
            Get evaluation metrics of one task of ED from support set and query set

            Parameters
            ----------
            support_set : pandas.DataFrame
                Support set.
            support_labels : numpy.ndarray
                Support set labels.
            query_set : pandas.DataFrame
                Query set.
            query_labels : numpy.ndarray
                Query set labels.

            Returns
            -------
            acc : float
                Accuracy of task
            prec : float
                Macro-averaged precision of task
            rec : float
                Macro-averaged recall of task
            f1 : float
                Macro-averaged F1-score of task
        """

        preds = []
        for query in query_set.values:
            query_vectors = np.broadcast_to(query, shape = support_set.shape)
            support_vectors = support_set.values

            # Find Euclidean distance between query and support and find label of nearest support sample
            pos = np.argmin(np.linalg.norm(query_vectors - support_vectors, axis = 1))
            preds.append(support_labels[pos])

        acc = accuracy_score(query_labels, preds)
        recall = recall_score(query_labels, preds, average = 'macro')
        precision = precision_score(query_labels, preds, average = 'macro')
        f1 = f1_score(query_labels, preds, average = 'macro')
            
        return acc, recall, precision, f1

    def get_metrics_k_shot_ED(self, X, y, k, q, n_tasks):
        """
            Get evaluation metrics of k-shot tasks of ED from a dataset

            Parameters
            ----------
            X : pandas.DataFrame
                Input data.
            y : pandas.Series
                Input labels.
            k : int
                Number of samples per class of support set.
            q : int
                Number of samples per class of query set.
            n_tasks : int
                Number of tasks.

            Returns
            -------
            accuracies : numpy.ndarray
                List of accuracies of each k-shot task of size (n_tasks, )
            recalls : numpy.ndarray
                List of macro-averaged recalls of each k-shot task of size (n_tasks, )
            precisions : numpy.ndarray
                List of macro-averaged precisions of each k-shot task of size (n_tasks, )
            f1s : numpy.ndarray
                List of macro-averaged F1-scores of each k-shot task of size (n_tasks, )

        """
        accuracies = []
        recalls = []
        precisions = []
        f1s = []
        for i in tqdm.tqdm(range(n_tasks)):
            # Sample k-shot task
            support_set, support_labels, query_set, query_labels = utils.sample_k_shot_task(X, y, k, q, seed_state_support=i, seed_state_query= n_tasks + i)

            # Get the evaluation metrics of the k-shot task
            acc, recall, precision, f1= self.get_metrics_of_task_ED(support_set, support_labels, query_set, query_labels)

            accuracies.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        return accuracies, recalls, precisions, f1s

    def print_metrics_ED(self, X_test, y_test, k_min, k_max, num_tasks, q):
        """
            Print evaluation metrics of k-shot tasks of ED from a dataset

            Parameters
            ----------
            X_test : pandas.DataFrame
                Input data.
            y_test : pandas.Series
                Input labels.
            k_min : int
                Minimum number of samples per class of support set.
            k_max : int
                Maximum number of samples per class of support set.
            num_tasks : int
                Number of ED tasks to run.
            q : int
                Number of samples per class of query set.

        """

        for i in range(k_min, k_max+1):
            accuracies, recalls, precisions, f1s= self.get_metrics_k_shot_ED(X_test, y_test, k = i, q = q, n_tasks= num_tasks)
            print('For %d shot, Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f '%(i, np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)))
            
            self.ED_accs.append(np.mean(accuracies))
            self.ED_recalls.append(np.mean(recalls))
            self.ED_precisions.append(np.mean(precisions))
            self.ED_F1s.append(np.mean(f1s))

        