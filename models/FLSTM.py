import numpy as np
import pandas as pd
import tqdm
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import utils
import config
import tensorflow as tf

class FLSTM_Model:
    def __init__(self, batch_size, epochs, verbose, output_dir):
        """
            Initialize the LSTM-FCN model

            Parameters
            ----------
            batch_size : int
                Size of batch to train the model
            epochs : int
                Number of epochs to train the model
            verbose : int
                Verbosity level
            output_dir : str
                Output directory to store model weights
            
        """
        self.model = self.build_fcnlstm_model()
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights_path = os.path.join(output_dir, 'lstmfcn.hdf5')
        self.verbose = verbose
        
        self.FLSTM_accs = []
        self.FLSTM_recalls = []
        self.FLSTM_precisions = []
        self.FLSTM_F1s = []
    
    def build_fcnlstm_model(self):
        """
            Build the LSTM-FCN model according the following paper:
            https://arxiv.org/abs/1709.05206
        """
    
        ip = tf.keras.layers.Input(shape=(1, config.PAD_SIZE))

        x = tf.keras.layers.LSTM(8)(ip)
        x = tf.keras.layers.Dropout(0.8)(x)

        y = tf.keras.layers.Permute((2, 1))(ip)
        y = tf.keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation('relu')(y)

        y = tf.keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation('relu')(y)

        y = tf.keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Activation('relu')(y)

        y = tf.keras.layers.GlobalAveragePooling1D()(y)

        x = tf.keras.layers.concatenate([x, y])

        out = tf.keras.layers.Dense(5, activation='softmax')(x)

        model = tf.keras.Model(ip, out)

        model.compile(loss = 'sparse_categorical_crossentropy', 
                      optimizer = tf.keras.optimizers.Adam(1e-4),
                      metrics = ['accuracy'])
        #print(model.summary())

        return model

    def get_metrics_of_task_FLSTM(self, support_set, support_labels, query_set, query_labels):
        """
            Get evaluation metrics of one task of LSTM-FCN from support set and query set

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

        tf.keras.backend.clear_session()

        X_train = np.expand_dims(support_set.values, axis = 1)
        X_test = np.expand_dims(query_set.values, axis = 1)
        y_train = support_labels
        y_test = query_labels

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.weights_path, 
                                                            monitor = 'val_accuracy',
                                                            save_best_only = True)
        
        # Fit LSTM-FCN model
        self.model.fit(
            X_train, y_train, 
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            validation_data = (X_test, y_test),
            verbose = self.verbose,
            callbacks = [model_checkpoint]
        )
        # Load best weights
        self.model.load_weights(self.weights_path)
        # Obtain predictions
        ypredtest = np.argmax(self.model.predict(X_test) , axis = 1)
        
        accuracy = accuracy_score(y_test, ypredtest)
        recall = recall_score(y_test, ypredtest, average = 'macro')
        precision = precision_score(y_test, ypredtest, average = 'macro')
        f1 = f1_score(y_test, ypredtest, average = 'macro')

        return accuracy, recall, precision, f1

    def get_metrics_k_shot_FLSTM(self, X, y, k, q, n_tasks):
        """
            Get evaluation metrics of k-shot tasks of the FCN-LSTM from a dataset

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
            acc, recall, precision, f1= self.get_metrics_of_task_FLSTM(support_set, support_labels, query_set, query_labels)

            accuracies.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        return accuracies, recalls, precisions, f1s

    def print_metrics_FLSTM(self, X_test, y_test, k_min, k_max, num_tasks, q):
        """
            Print evaluation metrics of k-shot tasks of LSTM-FCN from a dataset

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
                Number of LSTM-FCN tasks to run.
            q : int
                Number of samples per class of query set.

        """

        for i in range(k_min, k_max+1):
            accuracies, recalls, precisions, f1s= self.get_metrics_k_shot_FLSTM(X_test, y_test, k = i, q = q, n_tasks= num_tasks)
            print('For %d shot, Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f '%(i, np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)))
            
            self.FLSTM_accs.append(np.mean(accuracies))
            self.FLSTM_recalls.append(np.mean(recalls))
            self.FLSTM_precisions.append(np.mean(precisions))
            self.FLSTM_F1s.append(np.mean(f1s))

        