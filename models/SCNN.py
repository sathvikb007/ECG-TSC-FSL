import numpy as np
import pandas as pd
import os
import tensorflow as tf
import utils
import config
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


class SCNN_Model:

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
        self.weights_path = os.path.join(output_dir, 'scnn.hdf5')
        self.verbose = verbose

        # Build the SCNN model
        self.siamese_network, self.embedding_module, self.relation_module = self.build_SCNN_model()

        if(retrain == False):
            # Load pretrained weights if retrain is set to False
            self.pretrained_weights = os.path.join(output_dir, 'SCNN_pretrained.hdf5')
            self.siamese_network.load_weights(self.pretrained_weights)

        self.SCNN_accs = []
        self.SCNN_recalls = []
        self.SCNN_precisions = []
        self.SCNN_F1s = []

    def build_SCNN_model(self, input_shape = (config.PAD_SIZE, 1)):
        """
            Build the Siamese Network architecture

            Parameters
            ----------
            input_shape: tuple
                shape of the input data
            embedding_size: int
                embedding size of the embedding module
            maxlen: int
                maximum length of the input sequences
            
            Returns
            -------
            siamese_network: keras.models.Model
                Keras model of the Siamese Network
            embedding_module: keras.models.Model
                Keras model of the embedding module
            relation_module: keras.models.Model
                Keras model of the relation module
        """
        tf.keras.backend.clear_session()
        left_input = tf.keras.layers.Input(input_shape)
        right_input = tf.keras.layers.Input(input_shape)
        inputs1 = tf.keras.layers.Input(shape=input_shape)

        # Convolutional Block 1
        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=7, activation='relu', kernel_initializer= 'he_uniform')(inputs1)
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
        # Defining the embedding module
        embedding_module = tf.keras.Model(inputs=inputs1, outputs=embedding, name = "Embedding_Module")
        
        input_l = tf.keras.layers.Input(shape = (self.embedding_size))
        input_r = tf.keras.layers.Input(shape = (self.embedding_size))      

        L1_layer = tf.keras.layers.Lambda(lambda tensors:tf.keras.backend.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([input_l, input_r])                             
        similarity = tf.keras.layers.Dense(1,activation='sigmoid')(L1_distance)     

        # Defning the relation module                        
        relation_module = tf.keras.models.Model(inputs = [input_l, input_r], outputs = similarity,
                                            name = 'Relation_Module')
        
        embedded_l = embedding_module(left_input)
        embedded_r = embedding_module(right_input)
        similarity_score = relation_module([embedded_l, embedded_r])

        # Defnining the entire Siamese Network
        siamese_network = tf.keras.Model(inputs=[left_input, right_input], outputs=similarity_score,
                                     name = "Siamese_Network")
                                     
                                    
        siamese_network.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= self.weights_path, monitor='val_loss', save_best_only=True, verbose = 1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

        self.callbacks = [model_checkpoint, early_stopping]

        return siamese_network, embedding_module, relation_module

    def fit(self, x_train, y_train, x_val, y_val):
        """
            Fit the model to the training data
            
            Parameters
            ----------
            x_train: numpy.ndarray
                training pairs   
            y_train: numpy.ndarray
                training labels      
            x_val: numpy.ndarray
                validation pairs
            y_val: numpy.ndarray
                validation labels
            batch_size: int
                batch size for training
            epochs: int
                number of epochs

            Returns
            -------
            hist: History
                History of training
        """

        hist = self.siamese_network.fit([ x_train[:,:,0], x_train[:,:,1] ], 
                                        y_train, 
                                        batch_size= self.batch_size, 
                                        epochs= self.epochs, 
                                        validation_data=([ x_val[:,:,0], x_val[:,:,1] ], y_val), 
                                        verbose= self.verbose,
                                        callbacks=self.callbacks
                                        )
        return hist

    def predict(self, x_train, y_train, x_val, y_val):
        """
            Predict the result of the model on the validation data
            
            Parameters
            ----------
            x_train: numpy.ndarray
                training pairs
                
            y_train: numpy.ndarray
                training labels
                
            x_val: numpy.ndarray
                validation pairs
                
            y_val: numpy.ndarray
                validation labels
                
            Returns
            -------
            y_pred_train: numpy.ndarray
                predictions on the training data
            
            y_pred_val: numpy.ndarray
                predictions on the validation data
            
        """
        if(self.retrain):
            self.siamese_network.load_weights(self.weights_path)
        else:
            self.siamese_network.load_weights(self.pretrained_weights)

        y_pred_train = self.siamese_network.predict([ x_train[:,:,0], x_train[:,:,1] ], verbose=1)
        y_pred_test = self.siamese_network.predict([ x_val[:,:,0], x_val[:,:,1] ], verbose=1)

        return y_pred_train, y_pred_test

    def sample_k_shot_task_SCNN(self, X, y, k, q, seed_state_support, seed_state_query):
        """
            Sample a FSL k-shot task for the SCNN model

            Parameters
            ----------
            X : pandas.DataFrame
                Input data.
            y : pandas.Series
                Input labels.
            k : int
                Number of samples per class.
            q : int
                Number of queries per class.
            seed_state_support : int
                Seed of support set.
            seed_state_query : int
                Seed of query set.
                
        """ 
        # Initialize the mean vectors, query set and query labels
        mean_vecs = np.zeros((y.nunique(), self.embedding_size))
        queries = np.empty((0, self.embedding_size))
        query_labels = np.empty(0)
        i = 0
        # Iterate over the unique labels 
        for label in np.sort(y.unique()):
            # Sample k instances of the current label for the support set, and post-pad with zeros
            samples = X[y == label].sample(n = k, replace = False, random_state = seed_state_support)
            samples_pad = tf.keras.preprocessing.sequence.pad_sequences(samples.values , maxlen = config.PAD_SIZE, 
                                                                        dtype = 'float64', padding = 'post')
            
            # Obtain embeddings of the padded samples of the support set - f(X)
            samples_embedding = self.embedding_module.predict(samples_pad)
            # Average the embeddings of the padded samples to get the mean vector of the class mu
            class_feature = np.mean(samples_embedding, axis = 0) 
            # Append the mean vector of the current class to the mean vectors matrix
            mean_vecs[i] = class_feature

            # Sample q instances of the current label for the query set, and post-pad with zeros
            query_samples = X[y == label].sample(n = q, replace = False, random_state = seed_state_query)
            query_pad = tf.keras.preprocessing.sequence.pad_sequences(query_samples.values , maxlen = config.PAD_SIZE, 
                                                                dtype = 'float64', padding = 'post')
            
            # Obtain embeddings of the padded samples of the query set - Q
            queries_embedding = self.embedding_module.predict(query_pad)
            # Append the embeddings of the query set to queries
            queries = np.concatenate((queries, queries_embedding), axis = 0)
            
            # Append the labels of the current class to the query labels
            query_labels = np.concatenate((query_labels, np.array([label]*q)))

            i += 1

        return mean_vecs, queries, query_labels

    def get_metrics_of_task_SCNN(self, mean_vecs, queries, query_labels):
        """
            Get evaluation metrics of one task of SCNN from support set and query set

            Parameters
            ----------
            mean_vecs: numpy.ndarray
                Mean vectors of each class of the support set
            queries: numpy.ndarray
                The query set
            query_labels: numpy.ndarray
                Labels of the queries

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
        for query in queries:
            mult_query = np.resize(query, (5,self.embedding_size)) 

            # Get similarity score g( mu, Q ) for each mean vector mu and and the query embedding Q
            similarity_scores = self.relation_module.predict([mean_vecs, mult_query])
            # Final prediction of the query is the index of the class with the highest similarity score
            preds.append(np.argmax(similarity_scores))
            
        acc = accuracy_score(query_labels, preds)
        recall = recall_score(query_labels, preds, average = 'macro')
        precision = precision_score(query_labels, preds, average = 'macro')
        f1 = f1_score(query_labels, preds, average = 'macro')
            
        return acc, recall, precision, f1

    def get_metrics_k_shot_SCNN(self, X, y, k, q, n_tasks):
        """
            Get evaluation metrics of k-shot tasks of the SCNN from a dataset

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
            mean_vecs, queries, query_labels = self.sample_k_shot_task_SCNN(X, y, k, q, seed_state_support= i, seed_state_query=  n_tasks + i)

            # Get the evaluation metrics of the k-shot task
            acc, recall, precision, f1 = self.get_metrics_of_task_SCNN(mean_vecs, queries, query_labels)

            accuracies.append(acc)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
        return accuracies, recalls, precisions, f1s


    def print_metrics_SCNN(self, X_test, y_test, k_min, k_max, num_tasks, q):
        """
            Print evaluation metrics of k-shot tasks of SCNN from a dataset

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
                Number of SCNN tasks to run.
            q : int
                Number of samples per class of query set.

        """
        for i in range(k_min, k_max+1):
            accuracies, recalls, precisions, f1s= self.get_metrics_k_shot_SCNN(X_test, y_test, k = i, q = q, n_tasks= num_tasks)
            print('For %d shot, Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f '%(i, np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1s)))

            self.SCNN_accs.append(np.mean(accuracies))
            self.SCNN_recalls.append(np.mean(recalls))
            self.SCNN_precisions.append(np.mean(precisions))
            self.SCNN_F1s.append(np.mean(f1s))