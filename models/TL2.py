import numpy as np
import pandas as pd
import os
import tensorflow as tf
import utils
import config
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Transfer_Learning_Model2:

    def __init__(self, retrain = False, output_dir = '../weights/', batch_size = None, epochs = None, verbose = None):
        """
            Initialize the SCNN model

            Parameters
            ----------
            retrain : bool
                Whether to retrain the model or not
            output_dir : str
                Output directory that stores model weights
            batch_size : int
                Size of batch to train the model
            epochs : int
                Number of epochs to train the model
            verbose : int
                Verbosity level      
            
        """
        self.retrain = retrain
        self.embedding_size = 128
        self.batch_size = batch_size
        self.epochs = epochs
        self.pretrained_weights_path = os.path.join(output_dir, 'tl2_pretrained_weights.hdf5')
        self.fine_tuned_weights_path = os.path.join(output_dir, 'tl2_fine_tuned_weights.hdf5')
        self.verbose = verbose

        # Build the SCNN model
        self.model = self.build_TL_model()

        
        if(self.retrain == 0):
            try:
                self.model.load_weights(self.pretrained_weights_path)
            except:
                print("Finding weights")

        self.TL2_accs = []
        self.TL2_recalls = []
        self.TL2_precisions = []
        self.TL2_F1s = []

    def build_TL_model(self, input_shape = (config.PAD_SIZE,1) ):
        """
            Build the transfer learning model with ECG200 as source dataset
        """
        tf.keras.backend.clear_session()
        input = tf.keras.layers.Input(shape=input_shape)

        # Convolutional Block 1
        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, activation='relu', kernel_initializer= 'he_uniform')(input)
        bn1 = tf.keras.layers.BatchNormalization()(conv1)
        drop1 = tf.keras.layers.Dropout(0.4)(bn1)
        pool1 = tf.keras.layers.MaxPooling1D(pool_size= 3)(drop1)
        # Convolutional Block 2
        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', kernel_initializer= 'he_uniform')(pool1)
        bn2 = tf.keras.layers.BatchNormalization()(conv2)
        drop2 = tf.keras.layers.Dropout(0.4)(bn2)
        pool2 = tf.keras.layers.MaxPooling1D(pool_size=3)(drop2)
        # Convolutional Block 3
        conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', kernel_initializer='he_uniform')(pool2)
        bn3 = tf.keras.layers.BatchNormalization()(conv3)
        drop3 = tf.keras.layers.Dropout(0.4)(bn3)
        pool3 = tf.keras.layers.MaxPooling1D(pool_size=2)(drop3)
        flat3 = tf.keras.layers.Flatten()(pool3)
        # Embedding layer
        embedding = tf.keras.layers.Dense(self.embedding_size, activation = 'relu')(flat3)
        # Output layer
        output = tf.keras.layers.Dense(5, activation = 'softmax')(embedding)

        model = tf.keras.models.Model(inputs=input, outputs=output)                                     
                                    
        model.compile(
            loss='sparse_categorical_crossentropy', 
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= self.pretrained_weights_path, monitor='val_loss', save_best_only=True, verbose = 0)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        self.callbacks = [model_checkpoint, early_stopping]

        return model

    def pretrain_model(self, x_train, y_train, x_val, y_val):
        """
            Pretrain the model with given training and validation data
        """

        hist = self.model.fit(
            x_train, 
            y_train, 
            batch_size= self.batch_size, 
            epochs= self.epochs, 
            validation_data=(x_val, y_val), 
            verbose= self.verbose,
            callbacks=self.callbacks
        )
                                        
        return hist

    def predict(self, x_train, y_train, x_val, y_val):
        
        if(self.retrain):
            self.siamese_network.load_weights(self.pretrained_weights_path)
        else:
            self.siamese_network.load_weights(self.pretrained_weights)

        y_pred_train = self.siamese_network.predict([ x_train[:,:,0], x_train[:,:,1] ], verbose=1)
        y_pred_test = self.siamese_network.predict([ x_val[:,:,0], x_val[:,:,1] ], verbose=1)

        return y_pred_train, y_pred_test

    def get_metrics_of_task_TL(self, support_set, support_labels, query_set, query_labels):
        """
            Get evaluation metrics of one task of TL model from support set and query set

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

        X_train = np.expand_dims(support_set.values, axis = -1)
        X_test = np.expand_dims(query_set.values, axis = -1)
        y_train = support_labels
        y_test = query_labels

        model_ckpt= tf.keras.callbacks.ModelCheckpoint(self.fine_tuned_weights_path, 
                                                            monitor = 'val_accuracy',
                                                            save_best_only = True)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = 30)
        
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )
        # Load the pretrained weights
        self.model.load_weights(self.pretrained_weights_path)
        # Finetune the model
        self.model.fit(
            X_train, y_train, 
            epochs = self.epochs, 
            batch_size = self.batch_size, 
            validation_data = (X_test, y_test),
            verbose = self.verbose,
            callbacks = [model_ckpt, early_stop]
        )
        # Load best weights
        self.model.load_weights(self.fine_tuned_weights_path)
        # Obtain predictions
        ypredtest = np.argmax(self.model.predict(X_test) , axis = 1)
        
        accuracy = accuracy_score(y_test, ypredtest)
        recall = recall_score(y_test, ypredtest, average = 'macro')
        precision = precision_score(y_test, ypredtest, average = 'macro')
        f1 = f1_score(y_test, ypredtest, average = 'macro')

        return accuracy, recall, precision, f1

    def get_metrics_k_shot_TL(self, X, y, k, q, n_tasks):
        """
            Get evaluation metrics of k-shot tasks of the transfer learning from a dataset

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
        for i in tqdm(range(n_tasks)):
            # Sample k-shot task
            support_set, support_labels, query_set, query_labels = utils.sample_k_shot_task(X, y, k, q, seed_state_support=i, seed_state_query= n_tasks + i)

            # Get the evaluation metrics of the k-shot task
            acc, recall, precision, f1= self.get_metrics_of_task_TL(support_set, support_labels, query_set, query_labels)

            accuracies.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        return accuracies, recalls, precisions, f1s

    def print_metrics_TL(self, X_test, y_test, k_min, k_max, num_tasks, q):
        """
            Print evaluation metrics of k-shot tasks of transfer learning model from a dataset

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
            accuracies, recalls, precisions, f1s= self.get_metrics_k_shot_TL(X_test, y_test, k = i, q = q, n_tasks= num_tasks)
            print('For %d shot, Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f '%(i, np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)))
            
            self.TL2_accs.append(np.mean(accuracies))
            self.TL2_recalls.append(np.mean(recalls))
            self.TL2_precisions.append(np.mean(precisions))
            self.TL2_F1s.append(np.mean(f1s))

        TL2_df = pd.DataFrame()
        TL2_df['TL2 Accuracy'] = self.TL2_accs
        TL2_df['TL2 Recall'] = self.TL2_recalls
        TL2_df['TL2 Precision'] = self.TL2_precisions
        TL2_df['TL2 F1'] = self.TL2_F1s