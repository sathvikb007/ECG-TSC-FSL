import numpy as np
import pandas as pd
import os
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from tqdm import tqdm
from time import time
import tensorflow as tf
import config
from sklearn.model_selection import train_test_split

def create_pairs_from_dataframe(df, pad_size):  
    """
        Creates pairs from a given dataframe. 
        For each label in the datafame all possible same value pairs are taken, with an equal number of different class pairs 
        Same class pairs are labelled 1, different class pairs are labelled 0

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe.
        pad_size : int
            Length to which the time series are padded.

        
        Returns 
        -------
        pairs : pandas.DataFrame
            Dataframe containing all pairs of time series.
        pairs_labels : numpy.ndarray
            Array containing the labels of the pairs.
    """
    start = time()
    pairs = np.empty((0, pad_size, 2))
    pair_labels = []

    for lbl in tqdm(df.Label.unique(), desc = 'labels'):
        # Find number of time series with label lbl (= n)
        single_label = df[df.Label == lbl]
        n = single_label.shape[0]


        """
            Forming same class pairs
            ------------------------
        """
        # Number of same label pairs p, given n (= n choose 2)
        p = int( (n * (n-1) ) / 2)
        #p = min(p, max_size)
        # Choose 2p time series with random seed
        t1 = single_label.sample(n = p, replace = True).iloc[:, 1:].values
        t2 = single_label.sample(n = p, replace = True).iloc[:, 1:].values
        # Rescale time series to [0, 1] range
        ts1 = ((t1-np.min(t1,axis=1).reshape(-1,1))/(np.max(t1, axis=1)-np.min(t1,axis=1)).reshape(-1,1)  )
        ts2 = ((t2-np.min(t2,axis=1).reshape(-1,1))/(np.max(t2, axis=1)-np.min(t2,axis=1)).reshape(-1,1)  )
        # Post-pad with zeros
        padded1 = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(ts1 , maxlen = pad_size, dtype = 'float64', padding= 'post'), axis = -1)
        padded2 = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(ts2 , maxlen = pad_size, dtype = 'float64', padding= 'post'), axis = -1)
        # Concatenate the padded time series to form a pair. p such same-label pairs are formed
        padded_pairs = np.concatenate((padded1, padded2), axis = 2)

        pairs = np.concatenate((pairs, padded_pairs), axis = 0)


        """
            Forming different class pairs
            -----------------------------
        """
        # Choose p time series with label lbl, p time series not having label lbl
        t1 = df[df.Label == (lbl)].sample(n = p, replace = True).iloc[:, 1:].values
        t2 = df[df.Label != (lbl)].sample(n = p, replace = True).iloc[:, 1:].values
        # Rescale time series to [0, 1] range
        ts1 = ((t1-np.min(t1,axis=1).reshape(-1,1))/(np.max(t1, axis=1)-np.min(t1,axis=1)).reshape(-1,1)  )
        ts2 = ((t2-np.min(t2,axis=1).reshape(-1,1))/(np.max(t2, axis=1)-np.min(t2,axis=1)).reshape(-1,1)  )
        # Post-pad with zeros
        padded1 = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(ts1 , maxlen = pad_size, dtype = 'float64', padding= 'post'), axis = -1)
        padded2 = np.expand_dims(tf.keras.preprocessing.sequence.pad_sequences(ts2 , maxlen = pad_size, dtype = 'float64', padding= 'post'), axis = -1)
        # Concatenate the padded time series to form a pair. p such different-label pairs are formed
        padded_pairs = np.concatenate((padded1, padded2), axis = 2)
        pairs = np.concatenate((pairs, padded_pairs), axis = 0)
        
        # Append the labels of the pairs
        pair_labels += ([1]*int(p) + [0]*int(p) )

    return pairs, np.array(pair_labels) 

def create_supervised_task(train_dataset_list, val_dataset_list, datasets_path, pad_size):
    """
        Creates a supervised task from the list of datasets, i.e.
        Creates Ptr and Pval from Dtr and Dval
        For each dataset in the list, pairs of time series are created.
        The pairs are labelled 1 if the time series are from the same class, 0 otherwise.
        
        Parameters
        ----------
        train_dataset_list : list
            List of datasets to be used for training.
        val_dataset_list : list
            List of datasets to be used for validation.
        datasets_path : str
            Path to the datasets.
        pad_size : int
            Length to which the time series are padded.
        
        Returns
        -------
        train_pairs : pandas.DataFrame
            Dataframe containing all pairs of time series for training.
        train_labels : numpy.ndarray
            Array containing the labels of the training pairs.
        val_pairs : pandas.DataFrame    
            Dataframe containing all pairs of time series for validation.
        val_labels : numpy.ndarray
            Array containing the labels of the validation pairs.
    """
    train_pairs = np.empty((0,pad_size, 2))
    train_labels = np.empty(0)
    
    val_pairs = np.empty((0,pad_size, 2))
    val_labels = np.empty(0)

    for dataset_name in train_dataset_list:
        folder_path = os.path.join(datasets_path, dataset_name)
        train_path = os.path.join(folder_path, dataset_name + "_TRAIN.tsv")
        test_path = os.path.join(folder_path, dataset_name + "_TEST.tsv")

        # Create training pairs from train file of dataset_name
        dataset_pairs, dataset_labels = create_pairs_from_dataframe(pd.read_table(train_path , 
                                                                                  sep = "\t", 
                                                                                  header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1))), pad_size) 
        
        # Append the pairs and labels 
        train_pairs = np.concatenate((train_pairs, dataset_pairs), axis = 0)
        train_labels = np.append(train_labels, dataset_labels)
        
        # Create training pairs from test file of dataset_name
        dataset_pairs, dataset_labels = create_pairs_from_dataframe(pd.read_table(test_path ,
                                                                                  sep = "\t", 
                                                                                  header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1))), pad_size)      
        
        # Append the pairs and labels
        train_pairs = np.concatenate((train_pairs, dataset_pairs), axis = 0)
        train_labels = np.append(train_labels, dataset_labels)

        print(f'Done making training pairs for {dataset_name}\n')
        
    for dataset_name in val_dataset_list:
        folder_path = os.path.join(datasets_path, dataset_name)
        train_path = os.path.join(folder_path, dataset_name + "_TRAIN.tsv")
        test_path = os.path.join(folder_path, dataset_name + "_TEST.tsv")

        # Create validation pairs from train file of dataset_name
        dataset_pairs, dataset_labels = create_pairs_from_dataframe(pd.read_table(train_path ,
                                                                                  sep = "\t", 
                                                                                  header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1))), pad_size)

        # Append the pairs and labels
        val_pairs = np.concatenate((val_pairs, dataset_pairs), axis = 0)
        val_labels = np.append(val_labels, dataset_labels)
        
        # Create validation pairs from test file of dataset_name
        dataset_pairs, dataset_labels = create_pairs_from_dataframe(pd.read_table(test_path ,
                                                                                  sep = "\t", 
                                                                                  header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1))),pad_size)

        # Append the pairs and labels
        val_pairs = np.concatenate((val_pairs, dataset_pairs), axis = 0)
        val_labels = np.append(val_labels, dataset_labels)

        print(f'Done making validation pairs for {dataset_name}\n')

    return train_pairs, train_labels, val_pairs, val_labels


def load_test_data():
    """
        Load MITBIH data for testing
    """
    df = pd.read_csv('../data/MITBIH/MITBIH_TEST.csv', header = None)
    X_test = df.drop(187, axis = 1)
    y_test = df[187]
    return X_test, y_test

def get_last_nonzero_index(arr):
    """
        Get the index of the last nonzero element in an array arr
    """
    return np.max(np.nonzero(arr))

def DTW_distance(a, b):
    """
        Compute DTW distance between two time series a and b
    """
    distance, _ = fastdtw(a, b, dist=euclidean)
    return distance

def sample_k_shot_task(X, y, k, q, seed_state_support, seed_state_query):
    """
        Sample an FSL k-shot task from a dataset for ED, DTW and LSTM-FCN

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
        
        Returns
        -------
        support_set : pandas.DataFrame
            Support set.
        support_labels : numpy.ndarray
            Support set labels.
        query_set : pandas.DataFrame
            Query set.
        query_labels : numpy.ndarray
            Query set labels.
    """

    support_set, query_set = pd.DataFrame(), pd.DataFrame()
    support_labels, query_labels = np.empty(0), np.empty(0)

    for label in np.sort(y.unique()):
        support_samples = X[y == label].sample(n = k, replace = False, random_state = seed_state_support)
        query_samples = X[y == label].sample(n = q, replace = False, random_state = seed_state_query)

        support_set = pd.concat([support_set, support_samples], axis = 0)
        query_set = pd.concat([query_set, query_samples], axis = 0)
        
        support_labels = np.concatenate((support_labels, np.array([label]*k)))
        query_labels = np.concatenate((query_labels, np.array([label]*q)))

    return support_set, support_labels, query_set, query_labels

def load_TL_data(pad_size = config.PAD_SIZE-140):
    """
    Load ECG5000 data for training
    """
    train_path = '../data/ECG5000/ECG5000_TRAIN.tsv'
    test_path = '../data/ECG5000/ECG5000_TEST.tsv'
    df_train = pd.read_table(train_path , sep = "\t", header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1)))
    df_test = pd.read_table(test_path , sep = "\t", header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1)))
    df = pd.concat([df_train, df_test], axis = 0)
    y = df['Label']-1
    X = df.drop('Label', axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train = np.expand_dims(np.pad(X_train, ((0,0), (pad_size,0)), 'constant', constant_values=0), axis=-1)
    X_val = np.expand_dims(np.pad(X_val, ((0,0), (pad_size,0)), 'constant', constant_values=0), axis=-1)
    # X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train , maxlen = pad_size, dtype = 'float64', padding= 'post')
    # X_val = tf.keras.preprocessing.sequence.pad_sequences(X_val , maxlen = pad_size, dtype = 'float64', padding= 'post')
    return X_train, y_train, X_val, y_val

def load_TL_data_2(pad_size = config.PAD_SIZE - 96):
    """
    Load ECG200 data for training
    """
    train_path = '../data/ECG200/ECG200_TRAIN.tsv'
    test_path = '../data/ECG200/ECG200_TEST.tsv'
    df_train = pd.read_table(train_path , sep = "\t", header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1)))
    df_test = pd.read_table(test_path , sep = "\t", header=None).rename(columns = lambda col: "Label" if col==0 else (str(col-1)))
    df = pd.concat([df_train, df_test], axis = 0)
    y = df['Label'].replace({-1:0})
    X = df.drop('Label', axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train = np.expand_dims(np.pad(X_train, ((0,0), (pad_size,0)), 'constant', constant_values=0), axis=-1)
    X_val = np.expand_dims(np.pad(X_val, ((0,0), (pad_size,0)), 'constant', constant_values=0), axis=-1)
    return X_train, y_train, X_val, y_val
